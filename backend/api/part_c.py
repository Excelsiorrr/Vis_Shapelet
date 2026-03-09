from __future__ import annotations

from fastapi import APIRouter

from backend.schemas.part_c import (
    MatchTensorResponse,
    PartCFromPartBRequest,
    PartCFromPartBResponse,
    PartCMatchRequest,
    PartCMetaResponse,
)
from backend.services.part_c_service import PartCService


service = PartCService()
router = APIRouter(prefix="/api/v1/part-c", tags=["Part C"])


@router.get("/datasets/{dataset_name}/meta", response_model=PartCMetaResponse)
def get_part_c_meta(dataset_name: str) -> PartCMetaResponse:
    return service.get_meta(dataset_name)


@router.post("/datasets/{dataset_name}/samples/{sample_id}/matches", response_model=MatchTensorResponse)
def get_match_tensor(dataset_name: str, sample_id: str, request: PartCMatchRequest) -> MatchTensorResponse:
    return service.get_match_tensor(dataset_name, sample_id, request)


@router.post("/navigation/from-part-b", response_model=PartCFromPartBResponse)
def from_part_b_navigation(request: PartCFromPartBRequest) -> PartCFromPartBResponse:
    return service.from_part_b_navigation(
        request.link,
        include_sequence=request.include_sequence,
        include_prediction=request.include_prediction,
        include_windows=request.include_windows,
    )
