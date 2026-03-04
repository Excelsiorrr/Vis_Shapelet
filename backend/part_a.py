from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


SPEC_VERSION = "v1"
SUPPORTED_DATASETS = ("mitecg", "mcce", "mcch", "mtce", "mtch")
DEFAULT_MARGIN_THRESHOLD = 0.1
FREQSHAPE_DATASETS = {"mcce", "mcch", "mtce", "mtch"}


class ApiWarning(BaseModel):
    code: str
    message: str


class DatasetListItem(BaseModel):
    dataset: str
    display_name: str
    dataset_file_supported: bool = False
    notes: list[str] = Field(default_factory=list)


class DatasetListResponse(BaseModel):
    spec_version: str = SPEC_VERSION
    datasets: list[DatasetListItem]


class DatasetMeta(BaseModel):
    dataset: str
    sampling_rate: str
    seq_len: int


class TrainingMeta(BaseModel):
    shapelet_num: int
    shapelet_len: int
    classifier_num_classes: int


class MetaResponse(BaseModel):
    spec_version: str = SPEC_VERSION
    dataset_meta: DatasetMeta
    training_meta: TrainingMeta
    train_split_mapping: SplitMappingNote
    test_split_mapping: SplitMappingNote
    warnings: list[ApiWarning] = Field(default_factory=list)


class ClusterProfile(BaseModel):
    cluster_id: int
    size: int
    sample_ids_preview: list[str]
    centroid_sequence: list[list[float]]
    median_sequence: list[list[float]]
    q25_sequence: list[list[float]]
    q75_sequence: list[list[float]]


class MetricSummary(BaseModel):
    acc: float
    f1: float
    auc: float | None


class MetricsResponse(BaseModel):
    spec_version: str = SPEC_VERSION
    test_metrics: MetricSummary
    sample_count: int
    class_distribution: dict[str, int]
    low_margin_count: int


class PredictionSummary(BaseModel):
    pred_class: int
    probs: list[float]
    margin: float


class TestSampleSummary(BaseModel):
    sample_id: str
    label: int | None
    prediction: PredictionSummary


class LowMarginSample(BaseModel):
    sample_id: str
    label: int | None
    pred_class: int
    probs: list[float]
    margin: float
    sequence: list[list[float]]


class LowMarginSampleSummary(BaseModel):
    sample_id: str
    label: int | None
    pred_class: int
    probs: list[float]
    margin: float
    sequence: list[list[float]]


class SplitMappingNote(BaseModel):
    requested_role: str
    actual_source: str


class ClustersResponse(BaseModel):
    spec_version: str = SPEC_VERSION
    dataset: str
    cluster_k: int
    train_cluster_profiles: list[ClusterProfile]
    warnings: list[ApiWarning] = Field(default_factory=list)


class LowMarginSamplesResponse(BaseModel):
    spec_version: str = SPEC_VERSION
    dataset: str
    margin_threshold: float
    total: int
    offset: int
    limit: int
    items: list[LowMarginSampleSummary]
    warnings: list[ApiWarning] = Field(default_factory=list)


class ClassSamplesResponse(BaseModel):
    spec_version: str = SPEC_VERSION
    dataset: str
    label: int
    total: int
    offset: int
    limit: int
    items: list[TestSampleSummary]
    warnings: list[ApiWarning] = Field(default_factory=list)


class SampleDetailResponse(BaseModel):
    spec_version: str = SPEC_VERSION
    dataset: str
    split: str
    sample_id: str
    label: int | None
    prediction: PredictionSummary
    sequence: list[list[float]]
    suggested_window_len: int
    warnings: list[ApiWarning] = Field(default_factory=list)


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
    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])
        self.inference_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _dataset_labels(self, dataset: Any) -> torch.Tensor:
        return _to_tensor(dataset.y).long()

    def _dataset_sequences(self, dataset: Any) -> torch.Tensor:
        # Normalize backend datasets to (N, T, D). Some loaders expose (T, N, D).
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
        sequences = self._dataset_sequences(dataset)
        return sequences[sample_id]

    @lru_cache(maxsize=len(SUPPORTED_DATASETS))
    def load_dataset_bundle(self, dataset_name: str) -> LoadedDataset:
        # Load and cache all backend resources needed by Part A for one dataset:
        # config, raw splits, classifier, shapelet metadata, and user-facing warnings.
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

        dataset_meta = DatasetMeta(
            dataset=dataset_name,
            sampling_rate=_resolve_sampling_rate(pipeline.args),
            seq_len=int(getattr(pipeline.args, "seq_len", train_dataset.X.shape[1])),
        )
        training_meta = TrainingMeta(
            shapelet_num=shapelet_num,
            shapelet_len=shapelet_len,
            classifier_num_classes=int(args.num_classes),
        )
        warnings = _build_warning_list(dataset_name)

        return LoadedDataset(
            dataset_name=dataset_name,
            args=pipeline.args,
            class_model=class_model,
            inference_device=self.inference_device,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            dataset_meta=dataset_meta,
            training_meta=training_meta,
            warnings=warnings,
            train_mapping=_split_mapping(dataset_name, "TRAIN"),
            test_mapping=_split_mapping(dataset_name, "TEST"),
        )

    def _predict_batch(self, model: Any, sequences: torch.Tensor, batch_size: int = 128) -> np.ndarray:
        # Run the classifier in small batches so device memory stays bounded.
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
                logits = model(src, times).detach().cpu().numpy()
                logits_parts.append(logits)

        return np.concatenate(logits_parts, axis=0)

    @lru_cache(maxsize=len(SUPPORTED_DATASETS))
    def _cached_test_predictions(self, dataset_name: str) -> CachedPredictions:
        bundle = self.load_dataset_bundle(dataset_name)
        test_x = self._dataset_sequences(bundle.test_dataset)
        test_y = self._dataset_labels(bundle.test_dataset)
        logits = self._predict_batch(bundle.class_model, test_x)
        y_true, preds, probs = self._summarize_predictions(test_y, logits)
        return CachedPredictions(y_true=y_true, preds=preds, probs=probs)

    def _summarize_predictions(self, labels: torch.Tensor, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Convert raw model output into frontend-friendly prediction pieces:
        # true labels, argmax predictions, and per-class probabilities.
        probs = _softmax_logits(logits)
        preds = probs.argmax(axis=1)
        y_true = labels.detach().cpu().numpy().astype(int)
        return y_true, preds, probs

    def _compute_metrics(self, y_true: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> MetricSummary:
        # Compute the overview metrics shown in Part A:
        # accuracy, macro F1, and AUC when the label distribution allows it.
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
        # Prepare training cluster summaries for the overview panel.
        # Sequences are z-normalized before KMeans so clustering focuses on shape.
        # The frontend view does not use PCA; instead it renders each cluster as the
        # raw-space centroid line, the original-series median center line, and the q25-q75 band.
        x = self._dataset_sequences(dataset)
        flattened = _z_normalize_for_clustering(x)
        n_clusters = max(1, min(cluster_k, flattened.shape[0]))
        kmeans = KMeans(n_clusters=n_clusters, random_state=2026, n_init=10)
        cluster_ids = kmeans.fit_predict(flattened)

        profiles: list[ClusterProfile] = []
        x_np = x.numpy()
        for cluster_id in range(n_clusters):
            member_indices = np.where(cluster_ids == cluster_id)[0]
            cluster_sequences = x_np[member_indices]
            centroid_sequence = np.mean(cluster_sequences, axis=0)
            median_sequence = np.median(cluster_sequences, axis=0)
            q25_sequence = np.quantile(cluster_sequences, 0.25, axis=0)
            q75_sequence = np.quantile(cluster_sequences, 0.75, axis=0)

            profiles.append(
                ClusterProfile(
                    cluster_id=int(cluster_id),
                    size=int(member_indices.size),
                    sample_ids_preview=[str(idx) for idx in member_indices[:8].tolist()],
                    centroid_sequence=centroid_sequence.tolist(),
                    median_sequence=median_sequence.tolist(),
                    q25_sequence=q25_sequence.tolist(),
                    q75_sequence=q75_sequence.tolist(),
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
        # Build two overview data blocks from test predictions:
        # samples grouped by true class, and a sorted list of low-margin samples that
        # are useful for a "model is uncertain here" panel in the frontend.
        grouped: dict[str, list[TestSampleSummary]] = {}
        low_margin: list[LowMarginSample] = []
        for idx in range(sequences.shape[0]):
            prob = probs[idx]
            margin = _margin_from_probs(prob)
            pred = int(preds[idx])
            label = int(labels[idx].item())
            summary = TestSampleSummary(
                sample_id=str(idx),
                label=label,
                prediction=PredictionSummary(
                    pred_class=pred,
                    probs=[float(v) for v in prob.tolist()],
                    margin=margin,
                ),
            )
            grouped.setdefault(str(label), []).append(summary)
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
        test_x = self._dataset_sequences(bundle.test_dataset)
        test_y = self._dataset_labels(bundle.test_dataset)
        _, low_margin = self._group_test_samples(test_x, test_y, cached.preds, cached.probs, margin_threshold)
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
        test_x = self._dataset_sequences(bundle.test_dataset)
        test_y = self._dataset_labels(bundle.test_dataset)
        _, low_margin = self._group_test_samples(test_x, test_y, cached.preds, cached.probs, margin_threshold)
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
        test_y = self._dataset_labels(bundle.test_dataset)
        items: list[TestSampleSummary] = []
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

    def get_sample_detail(self, dataset_name: str, split: str, sample_id: int) -> SampleDetailResponse:
        # Return the detail payload for one sample, including the raw sequence,
        # prediction summary, and the suggested initial window length for the UI.
        bundle = self.load_dataset_bundle(dataset_name)
        split_name = split.upper()
        if split_name not in {"TRAIN", "TEST"}:
            raise HTTPException(status_code=400, detail="split must be one of: train, test")
        dataset = bundle.train_dataset if split_name == "TRAIN" else bundle.test_dataset
        labels = self._dataset_labels(dataset)
        sample_count = int(labels.shape[0])
        if sample_id < 0 or sample_id >= sample_count:
            raise HTTPException(status_code=404, detail=f"sample_id out of range: {sample_id}")

        sequence = self._dataset_sequence_at(dataset, sample_id).unsqueeze(0)
        label = int(labels[sample_id].item())
        logits = self._predict_batch(bundle.class_model, sequence)[0]
        probs = _softmax_logits(logits[np.newaxis, :])[0]
        prediction = PredictionSummary(
            pred_class=int(probs.argmax()),
            probs=[float(v) for v in probs.tolist()],
            margin=_margin_from_probs(probs),
        )

        return SampleDetailResponse(
            dataset=bundle.dataset_name,
            split=split_name.lower(),
            sample_id=str(sample_id),
            label=label,
            prediction=prediction,
            sequence=_to_numpy_2d(sequence[0]).tolist(),
            suggested_window_len=bundle.training_meta.shapelet_len,
            warnings=bundle.warnings,
        )


service = PartAService()
router = APIRouter(prefix="/api/v1/part-a", tags=["Part A"])


@router.get("/datasets", response_model=DatasetListResponse)
def list_datasets() -> DatasetListResponse:
    # Initialize the Part A dataset selector.
    # This exposes the built-in datasets and attaches notes about current MVP limitations.
    items = []
    for dataset_name in SUPPORTED_DATASETS:
        notes = ["MVP only supports built-in datasets; `dataset_file` is not enabled."]
        if dataset_name in FREQSHAPE_DATASETS:
            notes.append("This dataset also exposes known split-mapping and num_classes documentation notes.")
        items.append(
            DatasetListItem(
                dataset=dataset_name,
                display_name=dataset_name.upper(),
                notes=notes,
            )
        )
    return DatasetListResponse(datasets=items)


@router.get("/datasets/{dataset_name}/meta", response_model=MetaResponse)
def get_dataset_meta(
    dataset_name: str,
) -> MetaResponse:
    return service.get_meta(dataset_name)


@router.get("/datasets/{dataset_name}/metrics", response_model=MetricsResponse)
def get_dataset_metrics(
    dataset_name: str,
    margin_threshold: float = Query(default=DEFAULT_MARGIN_THRESHOLD, ge=0.0, le=1.0),
) -> MetricsResponse:
    return service.get_metrics(dataset_name, margin_threshold)


@router.get("/datasets/{dataset_name}/clusters", response_model=ClustersResponse)
def get_dataset_clusters(
    dataset_name: str,
    cluster_k: int = Query(default=4, ge=1, le=20),
) -> ClustersResponse:
    return service.get_clusters(dataset_name, cluster_k)


@router.get("/datasets/{dataset_name}/samples/low-margin", response_model=LowMarginSamplesResponse)
def get_low_margin_samples(
    dataset_name: str,
    threshold: float = Query(default=DEFAULT_MARGIN_THRESHOLD, ge=0.0, le=1.0),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> LowMarginSamplesResponse:
    return service.list_low_margin_samples(dataset_name, threshold, offset, limit)


@router.get("/datasets/{dataset_name}/samples", response_model=ClassSamplesResponse)
def get_class_samples(
    dataset_name: str,
    label: int = Query(..., ge=0),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> ClassSamplesResponse:
    return service.list_class_samples(dataset_name, label, offset, limit)


@router.get("/datasets/{dataset_name}/samples/{sample_id}", response_model=SampleDetailResponse)
def get_sample_detail(
    dataset_name: str,
    sample_id: int,
    split: str = Query(default="test"),
) -> SampleDetailResponse:
    # HTTP entrypoint for sample drill-down.
    # The frontend typically calls this after a user clicks a sample in a list or chart.
    return service.get_sample_detail(dataset_name, split, sample_id)


def create_app() -> FastAPI:
    # Create the FastAPI application and register the Part A router under /api/v1/part-a.
    app = FastAPI(title="ShapeX Part A API", version=SPEC_VERSION)
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def part_a_debug_home() -> FileResponse:
        return FileResponse(Path(__file__).with_name("part_a_debug.html"))

    @app.get("/part-a-debug", include_in_schema=False)
    def part_a_debug_page() -> FileResponse:
        return FileResponse(Path(__file__).with_name("part_a_debug.html"))

    return app


app = create_app()
