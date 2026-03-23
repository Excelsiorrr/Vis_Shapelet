from __future__ import annotations

from fastapi import APIRouter, Query

from backend.core.constants import DEFAULT_MARGIN_THRESHOLD, FREQSHAPE_DATASETS, SPEC_VERSION, SUPPORTED_DATASETS
from backend.schemas.part_a import (
    ClassSamplesResponse,
    ClustersResponse,
    DatasetListItem,
    DatasetListResponse,
    DepthProfileResponse,
    LowMarginSamplesResponse,
    MetaResponse,
    MetricsResponse,
    SampleDetailResponse,
)
from backend.services.part_a_service import PartAService


service = PartAService()
router = APIRouter(prefix="/api/v1/part-a", tags=["Part A"])


@router.get("/datasets", response_model=DatasetListResponse)
def list_datasets() -> DatasetListResponse:
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
def get_dataset_meta(dataset_name: str) -> MetaResponse:
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



@router.get("/datasets/{dataset_name}/samples/depth-profile", response_model=DepthProfileResponse)
def get_depth_profile_by_pred_class(
    dataset_name: str,
    pred_class: int = Query(..., ge=0),
    split: str = Query(default="test"),
) -> DepthProfileResponse:
    return service.get_depth_profile_by_pred_class(dataset_name, split, pred_class)

@router.get("/datasets/{dataset_name}/samples/{sample_id}", response_model=SampleDetailResponse)
def get_sample_detail(
    dataset_name: str,
    sample_id: int,
    split: str = Query(default="test"),
) -> SampleDetailResponse:
    return service.get_sample_detail(dataset_name, split, sample_id)

