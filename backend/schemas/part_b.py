from __future__ import annotations

from pydantic import BaseModel, Field

from backend.core.constants import SPEC_VERSION
from backend.schemas.part_a import ApiWarning


class HistogramDefault(BaseModel):
    mode: str
    bins: int
    density: bool


class ShapeletLibraryMetaResponse(BaseModel):
    spec_version: str = SPEC_VERSION
    dataset: str
    scope_default: str
    omega_default: float
    trigger_rule: str
    histogram_default: HistogramDefault
    warnings: list[ApiWarning] = Field(default_factory=list)


class ShapeletGalleryItem(BaseModel):
    shapelet_id: str
    shapelet_len: int
    ckpt_id: str
    prototype: list[list[float]]
    sample_ids_preview: list[str] = Field(default_factory=list)


class ShapeletGalleryListResponse(BaseModel):
    spec_version: str = SPEC_VERSION
    dataset: str
    total: int
    offset: int
    limit: int
    items: list[ShapeletGalleryItem]
    warnings: list[ApiWarning] = Field(default_factory=list)


class ShapeletDetailResponse(BaseModel):
    spec_version: str = SPEC_VERSION
    dataset: str
    shapelet: ShapeletGalleryItem
    warnings: list[ApiWarning] = Field(default_factory=list)


class SupportSummary(BaseModel):
    triggered_samples: int
    total_samples: int
    min_support: int
    alpha: float


class ShapeletStatsSummaryResponse(BaseModel):
    spec_version: str = SPEC_VERSION
    dataset: str
    shapelet_id: str
    scope: str
    omega: float
    global_trigger_rate: float
    class_trigger_rate: dict[str, float]
    class_coverage: dict[str, float]
    lift: dict[str, float | None]
    support: SupportSummary
    warnings: list[ApiWarning] = Field(default_factory=list)


class ShapeletHistogramResponse(BaseModel):
    spec_version: str = SPEC_VERSION
    dataset: str
    scope: str
    hist_mode: str
    shapelet_id: str | None
    bins: int
    density: bool
    range: list[float]
    counts: list[float]
    bin_edges: list[float]
    warnings: list[ApiWarning] = Field(default_factory=list)


class ShapeletSegmentPreviewItem(BaseModel):
    start: int
    end: int
    length: int
    peak_value: float


class ShapeletSegmentPreviewResponse(BaseModel):
    spec_version: str = SPEC_VERSION
    dataset: str
    shapelet_id: str
    scope: str
    seg_threshold: float
    signal_type: str
    time_axis: list[int]
    curve: list[float]
    segments: list[ShapeletSegmentPreviewItem]
    segment_count: int
    covered_ratio: float
    longest_segment: int
    warnings: list[ApiWarning] = Field(default_factory=list)


class ShapeletEvidenceMatchItem(BaseModel):
    sample_id: str
    rank: int
    label: int | None
    pred_class: int | None
    margin: float | None
    peak_t: int
    peak_activation: float
    t_start: int
    t_end: int
    raw_window: list[list[float]]
    activation_window: list[float]


class ShapeletEvidenceTopMatchesResponse(BaseModel):
    spec_version: str = SPEC_VERSION
    dataset: str
    shapelet_id: str
    scope: str
    shapelet_length: int
    rank_metric: str
    limit: int
    items: list[ShapeletEvidenceMatchItem]
    warnings: list[ApiWarning] = Field(default_factory=list)


class ShapeletMatrixSummaryResponse(BaseModel):
    spec_version: str = SPEC_VERSION
    dataset: str
    shapelet_id: str
    scope: str
    omega: float
    time_bins: int
    row_bins: int
    aggregation: str
    normalization: str
    sort_mode: str
    time_edges: list[int]
    row_sizes: list[int]
    representative_sample_ids: list[str]
    representative_curves: list[list[float]]
    matrix: list[list[float]]
    summary_median: list[float]
    summary_q25: list[float]
    summary_q75: list[float]
    exceed_ratio: list[float]
    value_range: list[float]
    warnings: list[ApiWarning] = Field(default_factory=list)


class ShapeletMatrixCellMember(BaseModel):
    sample_id: str
    label: int | None
    pred_class: int | None
    margin: float | None
    activation_curve: list[float]
    sequence_curve: list[float]
    activation_peak: float


class ShapeletMatrixCellDetailResponse(BaseModel):
    spec_version: str = SPEC_VERSION
    dataset: str
    shapelet_id: str
    scope: str
    omega: float
    row_index: int
    time_bin_index: int
    time_start: int
    time_end: int
    row_size: int
    cell_value: float
    exceed_ratio: float
    sample_ids: list[str]
    activation_median: list[float]
    sequence_median: list[float]
    members: list[ShapeletMatrixCellMember]
    warnings: list[ApiWarning] = Field(default_factory=list)


class ShapeletClassStatsItem(BaseModel):
    class_id: int
    prior: float
    trigger_rate: float
    coverage: float
    lift: float | None


class ShapeletClassStatsResponse(BaseModel):
    spec_version: str = SPEC_VERSION
    dataset: str
    shapelet_id: str
    scope: str
    omega: float
    items: list[ShapeletClassStatsItem]
    warnings: list[ApiWarning] = Field(default_factory=list)


class TopHitSampleItem(BaseModel):
    sample_id: str
    trigger_score: float
    rank: int
    label: int | None
    pred_class: int | None
    margin: float | None
    peak_t: int
    t_start: int
    t_end: int


class ShapeletTopHitsResponse(BaseModel):
    spec_version: str = SPEC_VERSION
    dataset: str
    shapelet_id: str
    scope: str
    omega: float
    class_id: int | None = None
    shapelet_length: int
    total: int
    offset: int
    limit: int
    rank_metric: str
    items: list[TopHitSampleItem]
    warnings: list[ApiWarning] = Field(default_factory=list)
