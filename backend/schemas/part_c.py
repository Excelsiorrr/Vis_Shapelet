from __future__ import annotations

from pydantic import BaseModel, Field

from backend.core.constants import SPEC_VERSION
from backend.schemas.part_a import ApiWarning, PredictionSummary


class PartCMetaResponse(BaseModel):
    spec_version: str = SPEC_VERSION
    dataset: str
    scope_default: str
    omega_default: float
    time_index_base: str = "0"
    peak_alignment: str = "center"
    match_score_semantics: str = "model_native_match_score"
    shapelet_len_source_priority: str = "match_response > part_b_link > shapelet_detail"
    warnings: list[ApiWarning] = Field(default_factory=list)


class PartBToCLink(BaseModel):
    dataset: str
    sample_id: str
    shapelet_id: str
    shapelet_len: int | None = None
    scope: str
    omega: float
    source_panel: str = "part_b"
    trigger_score: float | None = None
    rank: int | None = None
    rank_metric: str | None = None


class PartCMatchRequest(BaseModel):
    scope: str = "test"
    omega: float
    shapelet_ids: list[str] | None = None
    topk_shapelets: int | None = None
    pinned_shapelet_id: str | None = None
    include_sequence: bool = True
    include_prediction: bool = True
    include_windows: bool = True


class HighlightWindow(BaseModel):
    shapelet_id: str
    shapelet_len: int
    peak_t: int
    start: int
    end: int
    peak_score: float
    triggered: bool


class PinnedShapeletStatus(BaseModel):
    shapelet_id: str | None
    is_present_in_tensor: bool
    peak_t: int | None
    triggered: bool | None


class MatchParams(BaseModel):
    similarity_type: str
    shapelet_temperature: float
    normalize: str
    score_semantics: str = "model_native_match_score"


class MatchTensorResponse(BaseModel):
    spec_version: str = SPEC_VERSION
    dataset: str
    sample_id: str
    split: str
    scope: str
    omega: float
    shapelet_ids: list[str]
    shapelet_lens: list[int]
    I: list[list[float]]
    peak_t: list[int]
    windows: list[HighlightWindow] | None
    pinned_shapelet: PinnedShapeletStatus
    sequence: list[list[float]] | None
    prediction: PredictionSummary | None
    params: MatchParams
    warnings: list[ApiWarning] = Field(default_factory=list)


class PartCFromPartBRequest(BaseModel):
    link: PartBToCLink
    include_sequence: bool = True
    include_prediction: bool = True
    include_windows: bool = True


class PartCFromPartBResponse(BaseModel):
    spec_version: str = SPEC_VERSION
    link: PartBToCLink
    resolved_match_request: PartCMatchRequest
    match: MatchTensorResponse
    warnings: list[ApiWarning] = Field(default_factory=list)
