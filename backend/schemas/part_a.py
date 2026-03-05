from __future__ import annotations

from pydantic import BaseModel, Field

from backend.core.constants import SPEC_VERSION


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


class SplitMappingNote(BaseModel):
    requested_role: str
    actual_source: str


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
