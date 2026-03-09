from __future__ import annotations

from pydantic import BaseModel, Field

from backend.core.constants import SPEC_VERSION
from backend.schemas.part_a import ApiWarning


class PartEWhatIfRequest(BaseModel):
    shapelet_id: str
    t_start: int
    t_end: int
    scope: str = "test"
    omega: float
    baseline: str = "linear_interp"
    value_type: str = "prob"
    target_class: int | None = None
    seed: int | None = None
    include_perturbed_sequence: bool = False


class PartEWhatIfResponse(BaseModel):
    spec_version: str = SPEC_VERSION
    dataset: str
    sample_id: str
    shapelet_id: str
    t_start: int
    t_end: int
    scope: str
    omega: float
    baseline: str
    value_type: str
    target_class: int | None
    seed: int
    p_original: float
    p_whatif: float
    delta: float
    delta_target: float | None
    pred_class_original: int
    pred_class_whatif: int
    y_true: int | None
    perturbed_sequence: list[list[float]] | None
    warnings: list[ApiWarning] = Field(default_factory=list)


class PartCToELink(BaseModel):
    dataset: str
    sample_id: str
    shapelet_id: str
    t_start: int | None = None
    t_end: int | None = None
    scope: str
    omega: float
    source_panel: str = "part_c"
