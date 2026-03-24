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
    ShapeletMatrixCellDetailResponse,
    ShapeletMatrixCellMember,
    ShapeletMatrixSummaryResponse,
    ShapeletStatsSummaryResponse,
    ShapeletTopHitsResponse,
    SupportSummary,
    TopHitSampleItem,
)
from backend.services.part_a_service import PartAService

_SCOPE_VALUES = {"test", "train", "all"}
_HIST_MODE_VALUES = {"per_shapelet", "global"}
_RANK_METRIC_VALUES = {"max_i", "trigger_score"}
_MATRIX_AGGREGATION_VALUES = {"max", "mean", "median"}
_MATRIX_NORMALIZATION_VALUES = {"none", "per_sample_minmax"}
_MATRIX_SORT_VALUES = {"peak_position", "peak_value"}
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


@dataclass
class MatrixViewBundle:
    """Prepared matrix view reused by summary and cell-detail endpoints."""

    order: np.ndarray  # [N]
    matrix: np.ndarray  # [N, T]
    bucketed: np.ndarray  # [N, B]
    compressed: np.ndarray  # [R, B]
    time_edges: np.ndarray  # [B + 1]
    row_edges: np.ndarray  # [R + 1]
    row_sizes: list[int]
    representative_sample_ids: list[str]
    representative_curves: list[list[float]]
    exceed_ratio: list[float]
    summary_median: np.ndarray  # [B]
    summary_q25: np.ndarray  # [B]
    summary_q75: np.ndarray  # [B]
    value_range: list[float]


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

    def _validate_matrix_aggregation(self, aggregation: str) -> str:
        normalized = aggregation.lower().strip()
        if normalized not in _MATRIX_AGGREGATION_VALUES:
            raise HTTPException(
                status_code=400,
                detail=f"aggregation must be one of: {sorted(_MATRIX_AGGREGATION_VALUES)}",
            )
        return normalized

    def _validate_matrix_normalization(self, normalization: str) -> str:
        normalized = normalization.lower().strip()
        if normalized not in _MATRIX_NORMALIZATION_VALUES:
            raise HTTPException(
                status_code=400,
                detail=f"normalization must be one of: {sorted(_MATRIX_NORMALIZATION_VALUES)}",
            )
        return normalized

    def _validate_matrix_sort(self, sort_mode: str) -> str:
        normalized = sort_mode.lower().strip()
        if normalized not in _MATRIX_SORT_VALUES:
            raise HTTPException(
                status_code=400,
                detail=f"sort_mode must be one of: {sorted(_MATRIX_SORT_VALUES)}",
            )
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

    def _build_matrix_view(
        self,
        dataset_name: str,
        shapelet_id: str,
        scope: str,
        omega: float,
        time_bins: int,
        row_bins: int,
        aggregation: str,
        normalization: str,
        sort_mode: str,
    ) -> tuple[Any, list[ApiWarning], MatrixViewBundle, np.ndarray]:
        """Prepare sorted and compressed matrix artifacts shared by multiple Part B views."""

        bundle = self.part_a_service.load_dataset_bundle(dataset_name)
        normalized_scope = self._validate_scope(scope)
        normalized_omega = self._validate_omega(omega)
        normalized_aggregation = self._validate_matrix_aggregation(aggregation)
        normalized_normalization = self._validate_matrix_normalization(normalization)
        normalized_sort = self._validate_matrix_sort(sort_mode)

        data = self._cached_activations(dataset_name, normalized_scope)
        warnings = list(data.warnings)
        p_idx = self._parse_shapelet_index(shapelet_id, data.max_scores.shape[1])

        matrix = data.activations[:, :, p_idx].astype(np.float32)  # [N, T]
        sample_count, seq_len = matrix.shape
        if sample_count == 0 or seq_len == 0:
            raise HTTPException(status_code=500, detail="No activation matrix available for matrix summary")

        peak_positions = matrix.argmax(axis=1)
        peak_values = matrix.max(axis=1)

        if normalized_sort == "peak_position":
            order = np.lexsort((-peak_values, peak_positions))
        else:
            order = np.lexsort((peak_positions, -peak_values))
        matrix = matrix[order]

        if normalized_normalization == "per_sample_minmax":
            row_min = matrix.min(axis=1, keepdims=True)
            row_max = matrix.max(axis=1, keepdims=True)
            span = np.maximum(row_max - row_min, 1e-9)
            matrix = (matrix - row_min) / span

        effective_time_bins = max(1, min(int(time_bins), int(seq_len)))
        effective_row_bins = max(1, min(int(row_bins), int(sample_count)))

        time_edges = np.linspace(0, seq_len, num=effective_time_bins + 1, dtype=int)
        time_edges[-1] = seq_len
        bucketed_cols: list[np.ndarray] = []
        exceed_ratio: list[float] = []
        for start, end in zip(time_edges[:-1], time_edges[1:]):
            end = max(int(end), int(start) + 1)
            window = matrix[:, int(start):end]
            if normalized_aggregation == "max":
                col = window.max(axis=1)
            elif normalized_aggregation == "median":
                col = np.median(window, axis=1)
            else:
                col = window.mean(axis=1)
            bucketed_cols.append(col.astype(np.float32))
            exceed_ratio.append(float(np.mean(window >= normalized_omega)))

        bucketed = np.stack(bucketed_cols, axis=1)  # [N, B]

        row_edges = np.linspace(0, sample_count, num=effective_row_bins + 1, dtype=int)
        row_edges[-1] = sample_count
        compressed_rows: list[np.ndarray] = []
        row_sizes: list[int] = []
        representative_sample_ids: list[str] = []
        representative_curves: list[list[float]] = []
        sorted_peak_values = peak_values[order]
        for start, end in zip(row_edges[:-1], row_edges[1:]):
            start_int = int(start)
            end_int = max(int(end), start_int + 1)
            window = bucketed[start_int:end_int, :]
            compressed_rows.append(np.median(window, axis=0).astype(np.float32))
            row_sizes.append(int(end_int - start_int))
            row_peak_values = sorted_peak_values[start_int:end_int]
            local_best = int(np.argmax(row_peak_values))
            representative_idx = int(order[start_int + local_best])
            representative_sample_ids.append(str(representative_idx))
            representative_curves.append([float(v) for v in bucketed[start_int + local_best, :].tolist()])

        compressed = np.stack(compressed_rows, axis=0)  # [R, B]
        summary_median = np.median(bucketed, axis=0)
        summary_q25 = np.quantile(bucketed, 0.25, axis=0)
        summary_q75 = np.quantile(bucketed, 0.75, axis=0)

        return (
            bundle,
            warnings,
            MatrixViewBundle(
                order=order,
                matrix=matrix,
                bucketed=bucketed,
                compressed=compressed,
                time_edges=time_edges,
                row_edges=row_edges,
                row_sizes=row_sizes,
                representative_sample_ids=representative_sample_ids,
                representative_curves=representative_curves,
                exceed_ratio=exceed_ratio,
                summary_median=summary_median,
                summary_q25=summary_q25,
                summary_q75=summary_q75,
                value_range=[float(compressed.min()), float(compressed.max())],
            ),
            data.labels,
        )

    def get_matrix_summary(
        self,
        dataset_name: str,
        shapelet_id: str,
        scope: str,
        omega: float,
        time_bins: int,
        row_bins: int,
        aggregation: str,
        normalization: str,
        sort_mode: str,
    ) -> ShapeletMatrixSummaryResponse:
        """Return a compressed time x sample matrix summary for one shapelet."""
        (
            bundle,
            warnings,
            matrix_view,
            _,
        ) = self._build_matrix_view(
            dataset_name,
            shapelet_id,
            scope,
            omega,
            time_bins,
            row_bins,
            aggregation,
            normalization,
            sort_mode,
        )
        normalized_scope = self._validate_scope(scope)
        normalized_omega = self._validate_omega(omega)
        normalized_aggregation = self._validate_matrix_aggregation(aggregation)
        normalized_normalization = self._validate_matrix_normalization(normalization)
        normalized_sort = self._validate_matrix_sort(sort_mode)

        return ShapeletMatrixSummaryResponse(
            dataset=bundle.dataset_name,
            shapelet_id=shapelet_id,
            scope=normalized_scope,
            omega=normalized_omega,
            time_bins=int(matrix_view.bucketed.shape[1]),
            row_bins=int(matrix_view.compressed.shape[0]),
            aggregation=normalized_aggregation,
            normalization=normalized_normalization,
            sort_mode=normalized_sort,
            time_edges=[int(v) for v in matrix_view.time_edges.tolist()],
            row_sizes=matrix_view.row_sizes,
            representative_sample_ids=matrix_view.representative_sample_ids,
            representative_curves=matrix_view.representative_curves,
            matrix=[[float(v) for v in row.tolist()] for row in matrix_view.compressed],
            summary_median=[float(v) for v in matrix_view.summary_median.tolist()],
            summary_q25=[float(v) for v in matrix_view.summary_q25.tolist()],
            summary_q75=[float(v) for v in matrix_view.summary_q75.tolist()],
            exceed_ratio=matrix_view.exceed_ratio,
            value_range=matrix_view.value_range,
            warnings=warnings,
        )

    def get_matrix_cell_detail(
        self,
        dataset_name: str,
        shapelet_id: str,
        row_index: int,
        time_bin_index: int,
        scope: str,
        omega: float,
        time_bins: int,
        row_bins: int,
        aggregation: str,
        normalization: str,
        sort_mode: str,
    ) -> ShapeletMatrixCellDetailResponse:
        """Return all samples contained in one matrix cell plus aligned raw/activation curves."""

        (
            bundle,
            warnings,
            matrix_view,
            labels,
        ) = self._build_matrix_view(
            dataset_name,
            shapelet_id,
            scope,
            omega,
            time_bins,
            row_bins,
            aggregation,
            normalization,
            sort_mode,
        )
        normalized_scope = self._validate_scope(scope)
        normalized_omega = self._validate_omega(omega)

        row_count = int(matrix_view.compressed.shape[0])
        col_count = int(matrix_view.bucketed.shape[1])
        if row_index < 0 or row_index >= row_count:
            raise HTTPException(status_code=400, detail=f"row_index must be between 0 and {row_count - 1}")
        if time_bin_index < 0 or time_bin_index >= col_count:
            raise HTTPException(status_code=400, detail=f"time_bin_index must be between 0 and {col_count - 1}")

        row_start = int(matrix_view.row_edges[row_index])
        row_end = max(int(matrix_view.row_edges[row_index + 1]), row_start + 1)
        member_indices = matrix_view.order[row_start:row_end]
        if member_indices.size == 0:
            raise HTTPException(status_code=500, detail="Selected matrix cell does not contain any samples")

        x, _, _ = self._scope_sequences_and_labels(dataset_name, normalized_scope)
        sequence_np = x.detach().cpu().numpy()
        if sequence_np.ndim != 3:
            raise HTTPException(status_code=500, detail="Scoped sequence tensor must be 3D for cell detail")

        predictions = self._cached_predictions(dataset_name, normalized_scope)
        members: list[ShapeletMatrixCellMember] = []
        raw_curves: list[np.ndarray] = []
        activation_curves: list[np.ndarray] = []
        for local_offset, scoped_idx in enumerate(member_indices.tolist()):
            seq_curve = sequence_np[int(scoped_idx), :, 0].astype(np.float32)
            act_curve = matrix_view.matrix[row_start + local_offset, :].astype(np.float32)
            raw_curves.append(seq_curve)
            activation_curves.append(act_curve)
            pred_probs = predictions.probs[int(scoped_idx)] if int(scoped_idx) < len(predictions.probs) else np.array([])
            members.append(
                ShapeletMatrixCellMember(
                    sample_id=str(int(scoped_idx)),
                    label=int(labels[int(scoped_idx)]) if int(scoped_idx) < len(labels) else None,
                    pred_class=int(predictions.preds[int(scoped_idx)]) if int(scoped_idx) < len(predictions.preds) else None,
                    margin=_margin_from_probs(pred_probs) if pred_probs.size else None,
                    activation_curve=[float(v) for v in act_curve.tolist()],
                    sequence_curve=[float(v) for v in seq_curve.tolist()],
                    activation_peak=float(act_curve.max()) if act_curve.size else 0.0,
                )
            )

        raw_matrix = np.stack(raw_curves, axis=0)
        activation_matrix = np.stack(activation_curves, axis=0)
        time_start = int(matrix_view.time_edges[time_bin_index])
        time_end = max(int(matrix_view.time_edges[time_bin_index + 1]) - 1, time_start)

        return ShapeletMatrixCellDetailResponse(
            dataset=bundle.dataset_name,
            shapelet_id=shapelet_id,
            scope=normalized_scope,
            omega=normalized_omega,
            row_index=int(row_index),
            time_bin_index=int(time_bin_index),
            time_start=time_start,
            time_end=time_end,
            row_size=int(len(members)),
            cell_value=float(matrix_view.compressed[row_index, time_bin_index]),
            exceed_ratio=float(matrix_view.exceed_ratio[time_bin_index]),
            sample_ids=[member.sample_id for member in members],
            activation_median=[float(v) for v in np.median(activation_matrix, axis=0).tolist()],
            sequence_median=[float(v) for v in np.median(raw_matrix, axis=0).tolist()],
            members=members,
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
