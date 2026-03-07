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
