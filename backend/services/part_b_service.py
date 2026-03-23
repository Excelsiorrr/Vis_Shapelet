from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import HTTPException

from backend.core.constants import DEFAULT_EXPLAIN_OMEGA, FREQSHAPE_DATASETS, SUPPORTED_DATASETS
from backend.schemas.part_a import ApiWarning
from backend.schemas.part_b import (
    HistogramDefault,
    ShapeletClassStatsItem,
    ShapeletClassStatsResponse,
    ShapeletDetailResponse,
    ShapeletGalleryItem,
    ShapeletGalleryListResponse,
    ShapeletHistogramResponse,
    ShapeletLibraryMetaResponse,
    ShapeletStatsSummaryResponse,
    ShapeletTopHitsResponse,
    SupportSummary,
    TopHitSampleItem,
)
from backend.services.part_a_service import PartAService

_SCOPE_VALUES = {"test", "train", "all"}
_HIST_MODE_VALUES = {"per_shapelet", "global"}
_RANK_METRIC_VALUES = {"max_i", "trigger_score"}
_SHAPELET_ID_RE = re.compile(r"\d+")


@dataclass
class ActivationBundle:
    """Container for cached activation-related tensors used by Part B stats APIs."""

    activations: np.ndarray  # [N, T, P]
    labels: np.ndarray  # [N]
    max_scores: np.ndarray  # [N, P]
    warnings: list[ApiWarning]


@dataclass
class PredictionBundle:
    """Container for cached scoped class predictions reused by top-hits responses."""

    labels: np.ndarray  # [N]
    preds: np.ndarray  # [N]
    probs: np.ndarray  # [N, C]
    warnings: list[ApiWarning]


def _to_tensor(value: Any) -> torch.Tensor:
    """Convert input to a CPU tensor; detach if the input is already a tensor."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.tensor(value)


def _dataset_labels(dataset: Any) -> torch.Tensor:
    """Read dataset labels and normalize to int64 tensor on CPU."""

    return _to_tensor(dataset.y).long()


def _dataset_sequences(dataset: Any) -> torch.Tensor:
    """Read sequence tensor and align it to [N, T, D] against label count."""

    x = _to_tensor(dataset.X).float()
    labels = _dataset_labels(dataset)
    sample_count = int(labels.shape[0])
    if x.dim() != 3:
        raise ValueError(f"Expected 3D dataset tensor, got shape {tuple(x.shape)}")
    if x.shape[0] == sample_count:
        return x
    if x.shape[1] == sample_count:
        return x.transpose(0, 1).contiguous()
    raise ValueError(f"Unable to align dataset tensor shape {tuple(x.shape)} with labels shape {tuple(labels.shape)}")


def _safe_div(numerator: float, denominator: float) -> float:
    """Division helper that avoids NaN/inf by returning 0.0 when denominator is 0."""

    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _softmax_logits(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _margin_from_probs(probs: np.ndarray) -> float:
    if probs.size <= 1:
        return 1.0
    top2 = np.sort(probs)[-2:]
    return float(top2[-1] - top2[-2])


def _lift_value(
    n_pc: int,
    n_p_trig: int,
    n_c: int,
    n_total: int,
    n_classes: int,
    alpha: float,
) -> float:
    """Compute smoothed lift: P(y=c | triggered) / P(y=c)."""

    p_y_c = (n_c + alpha) / (n_total + n_classes * alpha)
    p_y_c_given_trig = (n_pc + alpha) / (n_p_trig + n_classes * alpha)
    return float(p_y_c_given_trig / p_y_c)


class PartBService:
    def __init__(self, project_root: Path | None = None) -> None:
        """Initialize shared services, defaults, and statistics stability thresholds."""

        self.project_root = Path(project_root or Path(__file__).resolve().parents[2])
        self.part_a_service = PartAService(self.project_root)
        self.inference_device = self.part_a_service.inference_device
        self.scope_default = "test"
        self.omega_default = float(DEFAULT_EXPLAIN_OMEGA)
        self.hist_bins_default = 50
        self.hist_density_default = True
        self.min_support = 20
        self.alpha = 1.0

    def _validate_scope(self, scope: str) -> str:
        """Normalize and validate scope; raise HTTP 400 for unsupported values."""

        normalized = scope.lower().strip()
        if normalized not in _SCOPE_VALUES:
            raise HTTPException(status_code=400, detail=f"scope must be one of: {sorted(_SCOPE_VALUES)}")
        return normalized

    def _validate_hist_mode(self, hist_mode: str) -> str:
        """Normalize and validate histogram mode; raise HTTP 400 on invalid mode."""

        normalized = hist_mode.lower().strip()
        if normalized not in _HIST_MODE_VALUES:
            raise HTTPException(status_code=400, detail=f"hist_mode must be one of: {sorted(_HIST_MODE_VALUES)}")
        return normalized

    def _validate_rank_metric(self, rank_metric: str) -> str:
        normalized = rank_metric.lower().strip()
        if normalized not in _RANK_METRIC_VALUES:
            raise HTTPException(status_code=400, detail=f"rank_metric must be one of: {sorted(_RANK_METRIC_VALUES)}")
        return normalized

    def _validate_omega(self, omega: float) -> float:
        if not math.isfinite(omega):
            raise HTTPException(status_code=400, detail="omega must be a finite float")
        return float(omega)

    def _parse_shapelet_index(self, shapelet_id: str, num_shapelets: int) -> int:
        """Extract trailing numeric index from shapelet_id and bounds-check it."""

        matches = _SHAPELET_ID_RE.findall(shapelet_id)
        if not matches:
            raise HTTPException(status_code=400, detail=f"Invalid shapelet_id: {shapelet_id}")
        shapelet_index = int(matches[-1])
        if shapelet_index < 0 or shapelet_index >= num_shapelets:
            raise HTTPException(status_code=404, detail=f"shapelet_id out of range: {shapelet_id}")
        return shapelet_index

    @lru_cache(maxsize=len(SUPPORTED_DATASETS))
    def _load_seg_model(self, dataset_name: str) -> Any:
        """Load and cache segmentation model for a dataset in eval mode."""

        dataset_name = dataset_name.lower()
        if dataset_name not in SUPPORTED_DATASETS:
            raise HTTPException(status_code=404, detail=f"Unsupported dataset: {dataset_name}")

        from omegaconf import OmegaConf

        from exp_saliency_general import ShapeXPipline

        config = OmegaConf.create(
            {
                "base": {"root_dir": str(self.project_root)},
                "dataset": {"name": dataset_name, "meta_dataset": "default"},
            }
        )
        pipeline = ShapeXPipline(config)
        pipeline.args.root_path = str(self.project_root)
        pipeline.args.use_gpu = self.inference_device.startswith("cuda")
        pipeline.args.device = self.inference_device
        seg_model = pipeline.load_seg_model().to(self.inference_device)
        seg_model.eval()
        return seg_model

    def _scope_sequences_and_labels(self, dataset_name: str, scope: str) -> tuple[torch.Tensor, torch.Tensor, list[ApiWarning]]:
        """Return scoped sequences/labels and attach scope-related warning messages."""

        bundle = self.part_a_service.load_dataset_bundle(dataset_name)
        warnings = list(bundle.warnings)

        if scope == "train":
            x = _dataset_sequences(bundle.train_dataset)
            y = _dataset_labels(bundle.train_dataset)
        elif scope == "test":
            x = _dataset_sequences(bundle.test_dataset)
            y = _dataset_labels(bundle.test_dataset)
        else:
            # scope=all concatenates train and test for exploratory analysis only.
            train_x = _dataset_sequences(bundle.train_dataset)
            test_x = _dataset_sequences(bundle.test_dataset)
            train_y = _dataset_labels(bundle.train_dataset)
            test_y = _dataset_labels(bundle.test_dataset)
            x = torch.cat([train_x, test_x], dim=0)
            y = torch.cat([train_y, test_y], dim=0)
            warnings.append(
                ApiWarning(
                    code="SCOPE_ALL_EXPLORATORY_NOTE",
                    message="`scope=all` is exploratory. Use `scope=test` for default external interpretation.",
                )
            )

        if bundle.dataset_name in FREQSHAPE_DATASETS:
            # Keep split-mapping caveat visible for datasets with known mapping behavior.
            warnings.append(
                ApiWarning(
                    code="SCOPE_SPLIT_MAPPING_NOTE",
                    message=(
                        "This dataset follows the current split-mapping behavior from the underlying loader. "
                        "Interpret train/test scope with the documented mapping notes."
                    ),
                )
            )
        return x, y, warnings

    def _predict_activations(self, seg_model: Any, sequences: torch.Tensor, batch_size: int = 64) -> np.ndarray:
        """Run model inference in batches and return activation tensor [N, T, P]."""

        x = sequences.detach().cpu().float()
        acts_parts: list[np.ndarray] = []
        model_device = next(seg_model.parameters()).device
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                end = min(start + batch_size, x.shape[0])
                chunk = x[start:end].to(model_device)
                # The model expects four sequence inputs; this pipeline reuses the same chunk.
                _, activations, _ = seg_model(chunk, chunk, chunk, chunk)
                acts_parts.append(activations.detach().cpu().numpy())
        return np.concatenate(acts_parts, axis=0)

    @lru_cache(maxsize=len(SUPPORTED_DATASETS) * 3)
    def _cached_activations(self, dataset_name: str, scope: str) -> ActivationBundle:
        """Compute and cache activations, labels, and per-sample max scores by scope."""

        normalized_scope = self._validate_scope(scope)
        x, y, warnings = self._scope_sequences_and_labels(dataset_name, normalized_scope)
        seg_model = self._load_seg_model(dataset_name)
        acts = self._predict_activations(seg_model, x)
        # Max over time gives one trigger score per sample and shapelet.
        max_scores = acts.max(axis=1)
        labels = y.detach().cpu().numpy().astype(int)
        return ActivationBundle(activations=acts, labels=labels, max_scores=max_scores, warnings=warnings)

    @lru_cache(maxsize=len(SUPPORTED_DATASETS) * 3)
    def _cached_predictions(self, dataset_name: str, scope: str) -> PredictionBundle:
        """Compute and cache class predictions for the scoped samples used in top-hits."""

        normalized_scope = self._validate_scope(scope)
        x, y, warnings = self._scope_sequences_and_labels(dataset_name, normalized_scope)
        bundle = self.part_a_service.load_dataset_bundle(dataset_name)
        logits = self.part_a_service._predict_batch(bundle.class_model, x)
        probs = _softmax_logits(logits)
        preds = probs.argmax(axis=1)
        labels = y.detach().cpu().numpy().astype(int)
        return PredictionBundle(labels=labels, preds=preds, probs=probs, warnings=warnings)

    @lru_cache(maxsize=len(SUPPORTED_DATASETS))
    def _cached_gallery_items(self, dataset_name: str) -> list[ShapeletGalleryItem]:
        """Build and cache static gallery metadata from model prototype vectors."""

        dataset_name = dataset_name.lower()
        bundle = self.part_a_service.load_dataset_bundle(dataset_name)
        seg_model = self._load_seg_model(dataset_name)
        proto = seg_model.prototype_vectors.detach().cpu().numpy()
        items: list[ShapeletGalleryItem] = []
        ckpt_id = f"{dataset_name}_shapex.pt"
        for idx in range(proto.shape[0]):
            items.append(
                ShapeletGalleryItem(
                    shapelet_id=f"s{idx:04d}",
                    shapelet_len=int(proto.shape[1]),
                    ckpt_id=ckpt_id,
                    prototype=proto[idx].tolist(),
                    sample_ids_preview=[],
                )
            )
        if bundle.training_meta.shapelet_num != len(items):
            # Keep the API resilient even if checkpoints diverge from expected metadata.
            return items[: bundle.training_meta.shapelet_num]
        return items

    def get_meta(self, dataset_name: str) -> ShapeletLibraryMetaResponse:
        """Return Part B defaults and dataset-level warnings for initial page load."""

        bundle = self.part_a_service.load_dataset_bundle(dataset_name)
        return ShapeletLibraryMetaResponse(
            dataset=bundle.dataset_name,
            scope_default=self.scope_default,
            omega_default=self.omega_default,
            trigger_rule="trigger_{n,p}(omega)=1{max_t I_{n,p,t}>=omega}",
            histogram_default=HistogramDefault(
                mode="per_shapelet",
                bins=self.hist_bins_default,
                density=self.hist_density_default,
            ),
            warnings=bundle.warnings,
        )

    def list_shapelets(self, dataset_name: str, offset: int, limit: int) -> ShapeletGalleryListResponse:
        """Return paginated static shapelet list for the gallery panel."""

        bundle = self.part_a_service.load_dataset_bundle(dataset_name)
        items = self._cached_gallery_items(dataset_name)
        return ShapeletGalleryListResponse(
            dataset=bundle.dataset_name,
            total=len(items),
            offset=offset,
            limit=limit,
            items=items[offset : offset + limit],
            warnings=bundle.warnings,
        )

    def get_shapelet_detail(self, dataset_name: str, shapelet_id: str) -> ShapeletDetailResponse:
        """Return one shapelet's static detail from cached gallery items."""

        bundle = self.part_a_service.load_dataset_bundle(dataset_name)
        items = self._cached_gallery_items(dataset_name)
        idx = self._parse_shapelet_index(shapelet_id, len(items))
        return ShapeletDetailResponse(dataset=bundle.dataset_name, shapelet=items[idx], warnings=bundle.warnings)

    def _summary_values(self, dataset_name: str, shapelet_id: str, scope: str, omega: float) -> tuple[dict[str, Any], list[ApiWarning]]:
        """Compute reusable per-shapelet statistics used by summary and class-stats endpoints."""

        data = self._cached_activations(dataset_name, scope)
        p_idx = self._parse_shapelet_index(shapelet_id, data.max_scores.shape[1])
        # Trigger rule: sample triggers when max_t I[n, t, p] >= omega.
        triggered = data.max_scores[:, p_idx] >= omega
        n_total = int(data.labels.shape[0])
        n_p_trig = int(triggered.sum())
        class_ids = sorted(np.unique(data.labels).tolist())
        n_classes = max(1, len(class_ids))

        class_counts: dict[int, int] = {class_id: int((data.labels == class_id).sum()) for class_id in class_ids}
        trig_class_counts: dict[int, int] = {
            class_id: int(np.logical_and(triggered, data.labels == class_id).sum()) for class_id in class_ids
        }

        class_trigger_rate: dict[str, float] = {}
        class_coverage: dict[str, float] = {}
        lift: dict[str, float | None] = {}
        warnings = list(data.warnings)

        for class_id in class_ids:
            # n_c: samples in class c, n_pc: triggered samples in class c.
            n_c = class_counts[class_id]
            n_pc = trig_class_counts[class_id]
            class_trigger_rate[str(class_id)] = _safe_div(n_pc, n_c)
            class_coverage[str(class_id)] = _safe_div(n_pc, n_c)
            if n_p_trig < self.min_support:
                lift[str(class_id)] = None
            else:
                lift[str(class_id)] = _lift_value(n_pc, n_p_trig, n_c, n_total, n_classes, self.alpha)

        if n_p_trig < self.min_support:
            warnings.append(
                ApiWarning(
                    code="LOW_SUPPORT_LIFT_UNSTABLE",
                    message=(
                        f"Triggered sample support ({n_p_trig}) is below min_support={self.min_support}; "
                        "lift is returned as null."
                    ),
                )
            )

        return (
            {
                "global_trigger_rate": _safe_div(n_p_trig, n_total),
                "class_trigger_rate": class_trigger_rate,
                "class_coverage": class_coverage,
                "lift": lift,
                "support": SupportSummary(
                    triggered_samples=n_p_trig,
                    total_samples=n_total,
                    min_support=self.min_support,
                    alpha=self.alpha,
                ),
                "class_ids": class_ids,
                "class_counts": class_counts,
                "trig_class_counts": trig_class_counts,
                "n_total": n_total,
                "n_p_trig": n_p_trig,
            },
            warnings,
        )

    def get_stats_summary(self, dataset_name: str, shapelet_id: str, scope: str, omega: float) -> ShapeletStatsSummaryResponse:
        """Return high-level per-shapelet statistics for cards/charts in the UI."""

        bundle = self.part_a_service.load_dataset_bundle(dataset_name)
        normalized_scope = self._validate_scope(scope)
        normalized_omega = self._validate_omega(omega)
        values, warnings = self._summary_values(dataset_name, shapelet_id, normalized_scope, normalized_omega)
        return ShapeletStatsSummaryResponse(
            dataset=bundle.dataset_name,
            shapelet_id=shapelet_id,
            scope=normalized_scope,
            omega=normalized_omega,
            global_trigger_rate=values["global_trigger_rate"],
            class_trigger_rate=values["class_trigger_rate"],
            class_coverage=values["class_coverage"],
            lift=values["lift"],
            support=values["support"],
            warnings=warnings,
        )

    def get_histogram(
        self,
        dataset_name: str,
        scope: str,
        hist_mode: str,
        shapelet_id: str | None,
        bins: int,
        density: bool,
        range_min: float | None,
        range_max: float | None,
    ) -> ShapeletHistogramResponse:
        """Return histogram data over activations, either per-shapelet or global."""

        bundle = self.part_a_service.load_dataset_bundle(dataset_name)
        normalized_scope = self._validate_scope(scope)
        normalized_mode = self._validate_hist_mode(hist_mode)
        data = self._cached_activations(dataset_name, normalized_scope)
        warnings = list(data.warnings)

        if normalized_mode == "per_shapelet":
            if not shapelet_id:
                raise HTTPException(status_code=400, detail="shapelet_id is required when hist_mode=per_shapelet")
            p_idx = self._parse_shapelet_index(shapelet_id, data.max_scores.shape[1])
            # Use all time positions for the selected shapelet across samples.
            values = data.activations[:, :, p_idx].reshape(-1)
        else:
            # Flatten all activations across sample/time/shapelet for global distribution.
            values = data.activations.reshape(-1)

        if values.size == 0:
            raise HTTPException(status_code=500, detail="No activation values available for histogram")

        hist_range = [float(values.min()), float(values.max())]
        if range_min is not None and range_max is not None and range_min < range_max:
            hist_range = [float(range_min), float(range_max)]
        counts, bin_edges = np.histogram(values, bins=bins, range=(hist_range[0], hist_range[1]), density=density)
        return ShapeletHistogramResponse(
            dataset=bundle.dataset_name,
            scope=normalized_scope,
            hist_mode=normalized_mode,
            shapelet_id=shapelet_id if normalized_mode == "per_shapelet" else None,
            bins=bins,
            density=density,
            range=hist_range,
            counts=[float(v) for v in counts.tolist()],
            bin_edges=[float(v) for v in bin_edges.tolist()],
            warnings=warnings,
        )

    def get_class_stats(self, dataset_name: str, shapelet_id: str, scope: str, omega: float) -> ShapeletClassStatsResponse:
        """Return per-class table rows for a shapelet at the selected threshold."""

        bundle = self.part_a_service.load_dataset_bundle(dataset_name)
        normalized_scope = self._validate_scope(scope)
        normalized_omega = self._validate_omega(omega)
        values, warnings = self._summary_values(dataset_name, shapelet_id, normalized_scope, normalized_omega)

        class_items: list[ShapeletClassStatsItem] = []
        for class_id in values["class_ids"]:
            n_c = values["class_counts"][class_id]
            n_pc = values["trig_class_counts"][class_id]
            class_items.append(
                ShapeletClassStatsItem(
                    class_id=int(class_id),
                    prior=_safe_div(n_c, values["n_total"]),
                    trigger_rate=_safe_div(n_pc, n_c),
                    coverage=_safe_div(n_pc, n_c),
                    lift=values["lift"][str(class_id)],
                )
            )
        return ShapeletClassStatsResponse(
            dataset=bundle.dataset_name,
            shapelet_id=shapelet_id,
            scope=normalized_scope,
            omega=normalized_omega,
            items=class_items,
            warnings=warnings,
        )

    def get_top_hits(
        self,
        dataset_name: str,
        shapelet_id: str,
        scope: str,
        omega: float,
        offset: int,
        limit: int,
        rank_metric: str,
    ) -> ShapeletTopHitsResponse:
        """Return triggered samples ranked by per-sample max activation for B->C linking."""

        bundle = self.part_a_service.load_dataset_bundle(dataset_name)
        normalized_scope = self._validate_scope(scope)
        normalized_omega = self._validate_omega(omega)
        normalized_rank_metric = self._validate_rank_metric(rank_metric)

        activations = self._cached_activations(dataset_name, normalized_scope)
        predictions = self._cached_predictions(dataset_name, normalized_scope)
        warnings = list(activations.warnings)

        p_idx = self._parse_shapelet_index(shapelet_id, activations.max_scores.shape[1])
        trigger_scores = activations.max_scores[:, p_idx]
        triggered_indices = [idx for idx, score in enumerate(trigger_scores.tolist()) if float(score) >= normalized_omega]
        ranked_indices = sorted(triggered_indices, key=lambda idx: (-float(trigger_scores[idx]), int(idx)))

        items: list[TopHitSampleItem] = []
        for rank, sample_idx in enumerate(ranked_indices[offset : offset + limit], start=offset + 1):
            probs = predictions.probs[sample_idx]
            items.append(
                TopHitSampleItem(
                    sample_id=str(sample_idx),
                    trigger_score=float(trigger_scores[sample_idx]),
                    rank=rank,
                    label=int(predictions.labels[sample_idx]) if sample_idx < len(predictions.labels) else None,
                    pred_class=int(predictions.preds[sample_idx]) if sample_idx < len(predictions.preds) else None,
                    margin=_margin_from_probs(probs) if sample_idx < len(predictions.probs) else None,
                )
            )

        return ShapeletTopHitsResponse(
            dataset=bundle.dataset_name,
            shapelet_id=shapelet_id,
            scope=normalized_scope,
            omega=normalized_omega,
            total=len(ranked_indices),
            offset=offset,
            limit=limit,
            rank_metric=normalized_rank_metric,
            items=items,
            warnings=warnings,
        )