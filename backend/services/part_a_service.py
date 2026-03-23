from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import HTTPException
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from backend.core.constants import FREQSHAPE_DATASETS, SUPPORTED_DATASETS
from backend.schemas.part_a import (
    ApiWarning,
    ClassSamplesResponse,
    ClusterProfile,
    ClustersResponse,
    DatasetMeta,
    DepthCentralRegion,
    DepthProfileResponse,
    DepthSample,
    DepthSummary,
    LowMarginSample,
    LowMarginSampleSummary,
    LowMarginSamplesResponse,
    MetaResponse,
    MetricsResponse,
    MetricSummary,
    PredictionSummary,
    SampleDetailResponse,
    SplitMappingNote,
    TestSampleSummary,
    TrainingMeta,
)


@dataclass
class LoadedDataset:
    dataset_name: str
    args: Any
    class_model: Any
    inference_device: str
    train_dataset: Any
    test_dataset: Any
    dataset_meta: DatasetMeta
    training_meta: TrainingMeta
    warnings: list[ApiWarning]
    train_mapping: SplitMappingNote
    test_mapping: SplitMappingNote


@dataclass
class CachedPredictions:
    y_true: np.ndarray
    preds: np.ndarray
    probs: np.ndarray


def _to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.tensor(value)


def _to_numpy_2d(seq: Any) -> np.ndarray:
    tensor = _to_tensor(seq).float()
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.dim() != 2:
        raise ValueError(f"Expected 2D sequence, got shape {tuple(tensor.shape)}")
    return tensor.numpy()


def _softmax_logits(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def _resolve_sampling_rate(args: Any) -> str:
    freq_value = str(getattr(args, "freq", "") or "").strip().lower()
    freq_map = {
        "h": "hour",
        "hour": "hour",
        "hours": "hour",
        "t": "min",
        "min": "min",
        "minute": "min",
        "minutes": "min",
        "d": "day",
        "day": "day",
        "days": "day",
        "s": "second",
        "sec": "second",
        "second": "second",
        "seconds": "second",
    }
    return freq_map.get(freq_value, "unknown")


def _z_normalize_for_clustering(sequences: torch.Tensor) -> np.ndarray:
    means = sequences.mean(dim=1, keepdim=True)
    stds = sequences.std(dim=1, keepdim=True, unbiased=False)
    stds = torch.where(stds < 1e-6, torch.ones_like(stds), stds)
    normalized = (sequences - means) / stds
    return normalized.reshape(normalized.shape[0], -1).numpy()


def _margin_from_probs(probs: np.ndarray) -> float:
    if probs.size == 1:
        return 1.0
    top2 = np.sort(probs)[-2:]
    return float(top2[-1] - top2[-2])

def _compute_soft_depth_against_mean(x: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute soft depth against mean curve, aligned with toy/fig.py."""
    if x.ndim != 2:
        raise ValueError(f"Expected [N,T], got shape {x.shape}")
    n = x.shape[0]
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, np.empty((0, x.shape[1]), dtype=float)

    mean_curve = np.mean(x, axis=0)
    sigma_t = np.std(x, axis=0)
    sigma_t = np.maximum(sigma_t, eps)

    w = np.exp(-((x - mean_curve[None, :]) ** 2) / (2.0 * (sigma_t[None, :] ** 2)))
    depth = np.mean(w, axis=1)
    return depth, mean_curve, sigma_t, w



def _z_normalize_rows(x: np.ndarray) -> np.ndarray:
    """Z-normalize each sample along time axis; input [N,T]."""
    if x.ndim != 2:
        raise ValueError(f"Expected [N,T], got shape {x.shape}")
    means = np.mean(x, axis=1, keepdims=True)
    stds = np.std(x, axis=1, keepdims=True)
    stds = np.where(stds < 1e-6, 1.0, stds)
    return (x - means) / stds


def _uniform_sample_indices(total: int, rate: float) -> np.ndarray:
    if total <= 0:
        return np.array([], dtype=int)
    count = max(1, int(np.ceil(total * rate)))
    if count >= total:
        return np.arange(total, dtype=int)
    return np.linspace(0, total - 1, num=count, dtype=int)


def _infer_num_classes_from_checkpoint(checkpoint_path: Path) -> int:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    head_weight = state_dict["mlp.2.weight"]
    return int(head_weight.shape[0])


def _extract_shapelet_meta(checkpoint_path: Path) -> tuple[int, int]:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "shapelet_layer.shapelets" in state_dict:
        weights = state_dict["shapelet_layer.shapelets"]
        if weights.dim() != 3:
            raise ValueError("Unexpected shapelet checkpoint shape")
        return int(weights.shape[0]), int(weights.shape[-1])
    if "prototype_vectors" in state_dict:
        weights = state_dict["prototype_vectors"]
        if weights.dim() != 3:
            raise ValueError("Unexpected prototype checkpoint shape")
        return int(weights.shape[0]), int(weights.shape[1])
    raise KeyError("No shapelet/prototype weights found in explainer checkpoint")


def _build_warning_list(dataset_name: str) -> list[ApiWarning]:
    warnings: list[ApiWarning] = [
        ApiWarning(
            code="DATASET_FILE_NOT_ENABLED",
            message="`dataset_file` is reserved in the PRD but is not implemented in the MVP backend. Use built-in datasets only.",
        )
    ]
    if dataset_name in FREQSHAPE_DATASETS:
        warnings.append(
            ApiWarning(
                code="SPLIT_MAPPING_KNOWN_ISSUE",
                message=(
                    "This dataset follows the current `get_saliency_data(return_dict=True)` mapping as-is. "
                    "`TRAIN` resolves to the underlying test split and `TEST` resolves to the underlying train split."
                ),
            )
        )
        warnings.append(
            ApiWarning(
                code="NUM_CLASSES_SOURCE_NOTE",
                message=(
                    "Classification output uses the checkpoint's actual class count (4). "
                    "The YAML config still exposes another choice (`num_classes: 2`) and is documented as a known inconsistency."
                ),
            )
        )
    return warnings


def _split_mapping(dataset_name: str, split_name: str) -> SplitMappingNote:
    if dataset_name in FREQSHAPE_DATASETS:
        actual_source = "underlying_test_split" if split_name == "TRAIN" else "underlying_train_split"
    else:
        actual_source = f"underlying_{split_name.lower()}_split"
    return SplitMappingNote(requested_role=split_name.lower(), actual_source=actual_source)


class PartAService:
    """Business layer for Part A.

    This class hides how ShapeX loads datasets and runs the classifier so the API
    layer only deals with HTTP parameters and typed responses.
    """

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2])
        self.inference_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _dataset_labels(self, dataset: Any) -> torch.Tensor:
        return _to_tensor(dataset.y).long()

    def _dataset_sequences(self, dataset: Any) -> torch.Tensor:
        x = _to_tensor(dataset.X).float()
        labels = self._dataset_labels(dataset)
        sample_count = int(labels.shape[0])

        if x.dim() != 3:
            raise ValueError(f"Expected 3D dataset tensor, got shape {tuple(x.shape)}")
        if x.shape[0] == sample_count:
            return x
        if x.shape[1] == sample_count:
            return x.transpose(0, 1).contiguous()
        raise ValueError(
            f"Unable to align dataset tensor shape {tuple(x.shape)} with labels shape {tuple(labels.shape)}"
        )

    def _dataset_sequence_at(self, dataset: Any, sample_id: int) -> torch.Tensor:
        return self._dataset_sequences(dataset)[sample_id]

    @lru_cache(maxsize=len(SUPPORTED_DATASETS))
    def load_dataset_bundle(self, dataset_name: str) -> LoadedDataset:
        dataset_name = dataset_name.lower()
        if dataset_name not in SUPPORTED_DATASETS:
            raise HTTPException(status_code=404, detail=f"Unsupported dataset: {dataset_name}")

        from omegaconf import OmegaConf

        from exp_saliency_general import ShapeXPipline, get_args
        from get_data import get_saliency_data

        config = OmegaConf.create(
            {
                "base": {"root_dir": str(self.project_root)},
                "dataset": {"name": dataset_name, "meta_dataset": "default"},
            }
        )
        args = get_args(config)
        args.root_path = str(self.project_root)
        args.use_gpu = self.inference_device.startswith("cuda")
        args.device = self.inference_device

        classifier_ckpt = self.project_root / "checkpoints" / "classification_models" / f"{dataset_name}_transformer.pt"
        if not classifier_ckpt.exists():
            raise HTTPException(status_code=500, detail=f"Missing classifier checkpoint: {classifier_ckpt}")
        args.num_classes = _infer_num_classes_from_checkpoint(classifier_ckpt)
        args.num_class = args.num_classes

        pipeline = ShapeXPipline(config)
        pipeline.args.root_path = str(self.project_root)
        pipeline.args.use_gpu = self.inference_device.startswith("cuda")
        pipeline.args.device = self.inference_device
        pipeline.args.num_classes = args.num_classes
        pipeline.args.num_class = args.num_classes

        data_dict = get_saliency_data(pipeline.args, return_dict=True)
        train_dataset = data_dict["TRAIN"]
        test_dataset = data_dict["TEST"]
        class_model = pipeline.get_class_model(train_dataset.X.cpu())
        class_model = class_model.to(self.inference_device)
        class_model.eval()

        explainer_ckpt = self.project_root / "checkpoints" / "explainer" / f"{dataset_name}_shapex.pt"
        if not explainer_ckpt.exists():
            raise HTTPException(status_code=500, detail=f"Missing explainer checkpoint: {explainer_ckpt}")
        shapelet_num, shapelet_len = _extract_shapelet_meta(explainer_ckpt)
        # Keep meta seq_len consistent with sample-detail sequence length.
        actual_seq_len = int(self._dataset_sequences(test_dataset).shape[1])

        dataset_meta = DatasetMeta(
            dataset=dataset_name,
            sampling_rate=_resolve_sampling_rate(pipeline.args),
            seq_len=actual_seq_len,
        )
        training_meta = TrainingMeta(
            shapelet_num=shapelet_num,
            shapelet_len=shapelet_len,
            classifier_num_classes=int(args.num_classes),
        )

        return LoadedDataset(
            dataset_name=dataset_name,
            args=pipeline.args,
            class_model=class_model,
            inference_device=self.inference_device,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            dataset_meta=dataset_meta,
            training_meta=training_meta,
            warnings=_build_warning_list(dataset_name),
            train_mapping=_split_mapping(dataset_name, "TRAIN"),
            test_mapping=_split_mapping(dataset_name, "TEST"),
        )

    def _predict_batch(self, model: Any, sequences: torch.Tensor, batch_size: int = 128) -> np.ndarray:
        x = sequences.detach().cpu().float()
        total, seq_len, _ = x.shape
        logits_parts: list[np.ndarray] = []
        model_device = next(model.parameters()).device

        with torch.no_grad():
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                chunk = x[start:end].to(model_device)
                chunk_batch = chunk.shape[0]
                times = torch.arange(1, seq_len + 1, dtype=torch.float32, device=model_device).unsqueeze(1).repeat(1, chunk_batch)
                src = chunk.transpose(0, 1)
                logits_parts.append(model(src, times).detach().cpu().numpy())

        return np.concatenate(logits_parts, axis=0)

    def _summarize_predictions(self, labels: torch.Tensor, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        probs = _softmax_logits(logits)
        preds = probs.argmax(axis=1)
        y_true = labels.detach().cpu().numpy().astype(int)
        return y_true, preds, probs

    @lru_cache(maxsize=len(SUPPORTED_DATASETS))
    def _cached_test_predictions(self, dataset_name: str) -> CachedPredictions:
        bundle = self.load_dataset_bundle(dataset_name)
        logits = self._predict_batch(bundle.class_model, self._dataset_sequences(bundle.test_dataset))
        y_true, preds, probs = self._summarize_predictions(self._dataset_labels(bundle.test_dataset), logits)
        return CachedPredictions(y_true=y_true, preds=preds, probs=probs)

    def _compute_metrics(self, y_true: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> MetricSummary:
        auc_value: float | None = None
        unique_labels = np.unique(y_true)
        if unique_labels.size >= 2:
            try:
                if probs.shape[1] == 2:
                    auc_value = float(roc_auc_score(y_true, probs[:, 1]))
                else:
                    eye = np.eye(probs.shape[1], dtype=float)
                    y_onehot = eye[y_true]
                    auc_value = float(roc_auc_score(y_onehot, probs, multi_class="ovr"))
            except ValueError:
                auc_value = None
        return MetricSummary(
            acc=float(accuracy_score(y_true, preds)),
            f1=float(f1_score(y_true, preds, average="macro", zero_division=0)),
            auc=auc_value,
        )

    def _build_training_cluster_profiles(self, dataset: Any, cluster_k: int) -> list[ClusterProfile]:
        x = self._dataset_sequences(dataset)
        flattened = _z_normalize_for_clustering(x)
        n_clusters = max(1, min(cluster_k, flattened.shape[0]))
        cluster_ids = KMeans(n_clusters=n_clusters, random_state=2026, n_init=10).fit_predict(flattened)

        profiles: list[ClusterProfile] = []
        x_np = x.numpy()
        for cluster_id in range(n_clusters):
            member_indices = np.where(cluster_ids == cluster_id)[0]
            cluster_sequences = x_np[member_indices]
            profiles.append(
                ClusterProfile(
                    cluster_id=int(cluster_id),
                    size=int(member_indices.size),
                    sample_ids_preview=[str(idx) for idx in member_indices[:8].tolist()],
                    centroid_sequence=np.mean(cluster_sequences, axis=0).tolist(),
                    median_sequence=np.median(cluster_sequences, axis=0).tolist(),
                    q25_sequence=np.quantile(cluster_sequences, 0.25, axis=0).tolist(),
                    q75_sequence=np.quantile(cluster_sequences, 0.75, axis=0).tolist(),
                )
            )
        return profiles

    def _group_test_samples(
        self,
        sequences: torch.Tensor,
        labels: torch.Tensor,
        preds: np.ndarray,
        probs: np.ndarray,
        margin_threshold: float,
    ) -> tuple[dict[str, list[TestSampleSummary]], list[LowMarginSample]]:
        grouped: dict[str, list[TestSampleSummary]] = {}
        low_margin: list[LowMarginSample] = []
        for idx in range(sequences.shape[0]):
            prob = probs[idx]
            margin = _margin_from_probs(prob)
            pred = int(preds[idx])
            label = int(labels[idx].item())
            grouped.setdefault(
                str(label),
                [],
            ).append(
                TestSampleSummary(
                    sample_id=str(idx),
                    label=label,
                    prediction=PredictionSummary(
                        pred_class=pred,
                        probs=[float(v) for v in prob.tolist()],
                        margin=margin,
                    ),
                )
            )
            if margin <= margin_threshold:
                low_margin.append(
                    LowMarginSample(
                        sample_id=str(idx),
                        label=label,
                        pred_class=pred,
                        probs=[float(v) for v in prob.tolist()],
                        margin=margin,
                        sequence=_to_numpy_2d(sequences[idx]).tolist(),
                    )
                )
        low_margin.sort(key=lambda item: item.margin)
        return grouped, low_margin

    @lru_cache(maxsize=len(SUPPORTED_DATASETS) * 4)
    def _cached_cluster_profiles(self, dataset_name: str, cluster_k: int) -> list[ClusterProfile]:
        bundle = self.load_dataset_bundle(dataset_name)
        return self._build_training_cluster_profiles(bundle.train_dataset, cluster_k)

    def get_meta(self, dataset_name: str) -> MetaResponse:
        bundle = self.load_dataset_bundle(dataset_name)
        return MetaResponse(
            dataset_meta=bundle.dataset_meta,
            training_meta=bundle.training_meta,
            train_split_mapping=bundle.train_mapping,
            test_split_mapping=bundle.test_mapping,
            warnings=bundle.warnings,
        )

    def get_metrics(self, dataset_name: str, margin_threshold: float) -> MetricsResponse:
        bundle = self.load_dataset_bundle(dataset_name)
        cached = self._cached_test_predictions(dataset_name)
        _, low_margin = self._group_test_samples(
            self._dataset_sequences(bundle.test_dataset),
            self._dataset_labels(bundle.test_dataset),
            cached.preds,
            cached.probs,
            margin_threshold,
        )
        class_distribution = {
            str(label): int(count)
            for label, count in zip(*np.unique(cached.y_true, return_counts=True), strict=False)
        }
        return MetricsResponse(
            test_metrics=self._compute_metrics(cached.y_true, cached.preds, cached.probs),
            sample_count=int(cached.y_true.shape[0]),
            class_distribution=class_distribution,
            low_margin_count=len(low_margin),
        )

    def get_clusters(self, dataset_name: str, cluster_k: int) -> ClustersResponse:
        bundle = self.load_dataset_bundle(dataset_name)
        return ClustersResponse(
            dataset=bundle.dataset_name,
            cluster_k=cluster_k,
            train_cluster_profiles=self._cached_cluster_profiles(dataset_name, cluster_k),
            warnings=bundle.warnings,
        )

    def list_low_margin_samples(
        self,
        dataset_name: str,
        margin_threshold: float,
        offset: int,
        limit: int,
    ) -> LowMarginSamplesResponse:
        bundle = self.load_dataset_bundle(dataset_name)
        cached = self._cached_test_predictions(dataset_name)
        _, low_margin = self._group_test_samples(
            self._dataset_sequences(bundle.test_dataset),
            self._dataset_labels(bundle.test_dataset),
            cached.preds,
            cached.probs,
            margin_threshold,
        )
        items = [
            LowMarginSampleSummary(
                sample_id=item.sample_id,
                label=item.label,
                pred_class=item.pred_class,
                probs=item.probs,
                margin=item.margin,
                sequence=item.sequence,
            )
            for item in low_margin[offset : offset + limit]
        ]
        return LowMarginSamplesResponse(
            dataset=bundle.dataset_name,
            margin_threshold=margin_threshold,
            total=len(low_margin),
            offset=offset,
            limit=limit,
            items=items,
            warnings=bundle.warnings,
        )

    def list_class_samples(self, dataset_name: str, label: int, offset: int, limit: int) -> ClassSamplesResponse:
        bundle = self.load_dataset_bundle(dataset_name)
        cached = self._cached_test_predictions(dataset_name)
        items: list[TestSampleSummary] = []
        test_y = self._dataset_labels(bundle.test_dataset)
        for idx, true_label in enumerate(test_y.tolist()):
            if int(true_label) != label:
                continue
            prob = cached.probs[idx]
            items.append(
                TestSampleSummary(
                    sample_id=str(idx),
                    label=int(true_label),
                    prediction=PredictionSummary(
                        pred_class=int(cached.preds[idx]),
                        probs=[float(v) for v in prob.tolist()],
                        margin=_margin_from_probs(prob),
                    ),
                )
            )
        return ClassSamplesResponse(
            dataset=bundle.dataset_name,
            label=label,
            total=len(items),
            offset=offset,
            limit=limit,
            items=items[offset : offset + limit],
            warnings=bundle.warnings,
        )

    @lru_cache(maxsize=len(SUPPORTED_DATASETS) * 16)
    def get_depth_profile_by_pred_class(
        self,
        dataset_name: str,
        split: str,
        pred_class: int,
    ) -> DepthProfileResponse:
        bundle = self.load_dataset_bundle(dataset_name)
        split_name = split.upper()
        if split_name != "TEST":
            raise HTTPException(status_code=400, detail="depth-profile currently supports split=test only")

        sequences = self._dataset_sequences(bundle.test_dataset)
        labels = self._dataset_labels(bundle.test_dataset).detach().cpu().numpy().astype(int)
        cached = self._cached_test_predictions(dataset_name)

        indices = np.where(cached.preds == int(pred_class))[0]
        if indices.size == 0:
            warnings = [
                *bundle.warnings,
                ApiWarning(
                    code="EMPTY_PRED_CLASS_GROUP",
                    message=f"No test samples are currently predicted as class {pred_class}.",
                ),
            ]
            return DepthProfileResponse(
                dataset=bundle.dataset_name,
                split=split_name.lower(),
                pred_class=int(pred_class),
                total=0,
                items=[],
                plot_items=[],
                plot_sample_rate=0.0,
                representative_sample_id=None,
                representative_sequence=[],
                mean_sequence=[],
                central_region=DepthCentralRegion(
                    lower_bound=[],
                    upper_bound=[],
                    threshold_depth=0.0,
                    central_ratio=0.5,
                    central_count=0,
                    band_mode="quantile",
                ),
                depth_summary=DepthSummary(min_depth=0.0, max_depth=0.0, median_depth=0.0, max_depth_sample_id=None),
                warnings=warnings,
            )

        idx_list = indices.tolist()
        seq_raw = np.stack([_to_numpy_2d(sequences[i])[:, 0] for i in idx_list], axis=0)
        seq_norm = _z_normalize_rows(seq_raw)
        depths, mean_curve, _sigma_t, _w = _compute_soft_depth_against_mean(seq_norm)

        sort_order = np.argsort(-depths)
        sorted_indices = [idx_list[i] for i in sort_order.tolist()]
        sorted_depths = depths[sort_order]
        sorted_seq_norm = seq_norm[sort_order]

        # Align with toy/fig.py panel-3 semantics:
        # choose top central_ratio samples by depth, then use quantile band.
        central_ratio = 0.5
        band_mode = "quantile"
        central_count = max(1, int(np.ceil(len(sorted_indices) * central_ratio)))
        central_x = sorted_seq_norm[:central_count]
        threshold_depth = float(sorted_depths[central_count - 1])
        if band_mode == "quantile":
            lower_bound = np.quantile(central_x, 0.25, axis=0).astype(float).tolist()
            upper_bound = np.quantile(central_x, 0.75, axis=0).astype(float).tolist()
        else:
            lower_bound = np.min(central_x, axis=0).astype(float).tolist()
            upper_bound = np.max(central_x, axis=0).astype(float).tolist()

        representative_idx = int(sorted_indices[0])
        representative_sequence = sorted_seq_norm[0].astype(float).tolist()
        mean_sequence = mean_curve.astype(float).tolist()

        items: list[DepthSample] = []
        for rank, sample_idx in enumerate(sorted_indices):
            prob = cached.probs[sample_idx]
            norm_seq_2d = [[float(v)] for v in sorted_seq_norm[rank].tolist()]
            items.append(
                DepthSample(
                    sample_id=str(sample_idx),
                    label=int(labels[sample_idx]),
                    prediction=PredictionSummary(
                        pred_class=int(cached.preds[sample_idx]),
                        probs=[float(v) for v in prob.tolist()],
                        margin=_margin_from_probs(prob),
                    ),
                    depth=float(sorted_depths[rank]),
                    sequence=norm_seq_2d,
                )
            )

        sample_rate = 0.1
        sample_pos = _uniform_sample_indices(len(items), sample_rate)
        plot_items = [items[int(i)] for i in sample_pos.tolist()]
        plot_sample_rate = float(len(plot_items) / len(items)) if items else 0.0

        return DepthProfileResponse(
            dataset=bundle.dataset_name,
            split=split_name.lower(),
            pred_class=int(pred_class),
            total=len(items),
            items=items,
            plot_items=plot_items,
            plot_sample_rate=plot_sample_rate,
            representative_sample_id=str(representative_idx),
            representative_sequence=representative_sequence,
            mean_sequence=mean_sequence,
            central_region=DepthCentralRegion(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                threshold_depth=threshold_depth,
                central_ratio=central_ratio,
                central_count=central_count,
                band_mode=band_mode,
            ),
            depth_summary=DepthSummary(
                min_depth=float(np.min(depths)),
                max_depth=float(np.max(depths)),
                median_depth=threshold_depth,
                max_depth_sample_id=str(representative_idx),
            ),
            warnings=bundle.warnings,
        )

    def get_sample_detail(self, dataset_name: str, split: str, sample_id: int) -> SampleDetailResponse:
        bundle = self.load_dataset_bundle(dataset_name)
        split_name = split.upper()
        if split_name not in {"TRAIN", "TEST"}:
            raise HTTPException(status_code=400, detail="split must be one of: train, test")
        dataset = bundle.train_dataset if split_name == "TRAIN" else bundle.test_dataset
        labels = self._dataset_labels(dataset)
        if sample_id < 0 or sample_id >= int(labels.shape[0]):
            raise HTTPException(status_code=404, detail=f"sample_id out of range: {sample_id}")

        sequence = self._dataset_sequence_at(dataset, sample_id).unsqueeze(0)
        probs = _softmax_logits(self._predict_batch(bundle.class_model, sequence)[0][np.newaxis, :])[0]
        return SampleDetailResponse(
            dataset=bundle.dataset_name,
            split=split_name.lower(),
            sample_id=str(sample_id),
            label=int(labels[sample_id].item()),
            prediction=PredictionSummary(
                pred_class=int(probs.argmax()),
                probs=[float(v) for v in probs.tolist()],
                margin=_margin_from_probs(probs),
            ),
            sequence=_to_numpy_2d(sequence[0]).tolist(),
            suggested_window_len=bundle.training_meta.shapelet_len,
            warnings=bundle.warnings,
        )





