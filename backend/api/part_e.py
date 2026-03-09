from __future__ import annotations

from fastapi import APIRouter

from backend.schemas.part_e import PartEWhatIfRequest, PartEWhatIfResponse
from backend.services.part_e_service import PartEService


service = PartEService()
router = APIRouter(prefix="/api/v1/part-e", tags=["Part E"])


@router.post("/datasets/{dataset_name}/samples/{sample_id}/whatif:evaluate", response_model=PartEWhatIfResponse)
def evaluate_whatif(
    dataset_name: str,
    sample_id: str,
    request: PartEWhatIfRequest,
) -> PartEWhatIfResponse:
    return service.evaluate_whatif(dataset_name, sample_id, request)
