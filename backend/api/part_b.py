from __future__ import annotations

from fastapi import APIRouter, Query

from backend.core.constants import DEFAULT_EXPLAIN_OMEGA
from backend.schemas.part_b import (
    ShapeletClassStatsResponse,
    ShapeletEvidenceTopMatchesResponse,
    ShapeletMatrixCellDetailResponse,
    ShapeletDetailResponse,
    ShapeletGalleryListResponse,
    ShapeletHistogramResponse,
    ShapeletLibraryMetaResponse,
    ShapeletSegmentPreviewResponse,
    ShapeletMatrixSummaryResponse,
    ShapeletStatsSummaryResponse,
    ShapeletTopHitsResponse,
)
from backend.services.part_b_service import PartBService


service = PartBService()
router = APIRouter(prefix="/api/v1/part-b", tags=["Part B"])


@router.get("/datasets/{dataset_name}/meta", response_model=ShapeletLibraryMetaResponse)
def get_part_b_meta(dataset_name: str) -> ShapeletLibraryMetaResponse:
    return service.get_meta(dataset_name)


@router.get("/datasets/{dataset_name}/shapelets", response_model=ShapeletGalleryListResponse)
def list_shapelets(
    dataset_name: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
) -> ShapeletGalleryListResponse:
    return service.list_shapelets(dataset_name, offset, limit)


@router.get("/datasets/{dataset_name}/shapelets/{shapelet_id}", response_model=ShapeletDetailResponse)
def get_shapelet_detail(dataset_name: str, shapelet_id: str) -> ShapeletDetailResponse:
    return service.get_shapelet_detail(dataset_name, shapelet_id)


@router.get("/datasets/{dataset_name}/shapelets/{shapelet_id}/stats/summary", response_model=ShapeletStatsSummaryResponse)
def get_shapelet_stats_summary(
    dataset_name: str,
    shapelet_id: str,
    scope: str = Query(default="test"),
    omega: float = Query(default=DEFAULT_EXPLAIN_OMEGA),
) -> ShapeletStatsSummaryResponse:
    return service.get_stats_summary(dataset_name, shapelet_id, scope, omega)


@router.get("/datasets/{dataset_name}/shapelets/stats/histogram", response_model=ShapeletHistogramResponse)
def get_shapelet_histogram(
    dataset_name: str,
    scope: str = Query(default="test"),
    hist_mode: str = Query(default="per_shapelet"),
    shapelet_id: str | None = Query(default=None),
    bins: int = Query(default=50, ge=5, le=500),
    density: bool = Query(default=True),
    range_min: float | None = Query(default=None),
    range_max: float | None = Query(default=None),
) -> ShapeletHistogramResponse:
    return service.get_histogram(dataset_name, scope, hist_mode, shapelet_id, bins, density, range_min, range_max)


@router.get(
    "/datasets/{dataset_name}/shapelets/{shapelet_id}/segments/preview",
    response_model=ShapeletSegmentPreviewResponse,
)
def get_shapelet_segment_preview(
    dataset_name: str,
    shapelet_id: str,
    scope: str = Query(default="test"),
    seg_threshold: float | None = Query(default=None),
) -> ShapeletSegmentPreviewResponse:
    return service.get_segment_preview(dataset_name, shapelet_id, scope, seg_threshold)


@router.get(
    "/datasets/{dataset_name}/shapelets/{shapelet_id}/evidence/top-matches",
    response_model=ShapeletEvidenceTopMatchesResponse,
)
def get_shapelet_evidence_top_matches(
    dataset_name: str,
    shapelet_id: str,
    scope: str = Query(default="test"),
    limit: int = Query(default=12, ge=1, le=50),
) -> ShapeletEvidenceTopMatchesResponse:
    return service.get_evidence_top_matches(dataset_name, shapelet_id, scope, limit)


@router.get(
    "/datasets/{dataset_name}/shapelets/{shapelet_id}/stats/matrix-summary",
    response_model=ShapeletMatrixSummaryResponse,
)
def get_shapelet_matrix_summary(
    dataset_name: str,
    shapelet_id: str,
    scope: str = Query(default="test"),
    omega: float = Query(default=DEFAULT_EXPLAIN_OMEGA),
    time_bins: int = Query(default=180, ge=24, le=400),
    row_bins: int = Query(default=120, ge=24, le=300),
    aggregation: str = Query(default="max"),
    normalization: str = Query(default="none"),
    sort_mode: str = Query(default="peak_position"),
) -> ShapeletMatrixSummaryResponse:
    return service.get_matrix_summary(
        dataset_name,
        shapelet_id,
        scope,
        omega,
        time_bins,
        row_bins,
        aggregation,
        normalization,
        sort_mode,
    )


@router.get(
    "/datasets/{dataset_name}/shapelets/{shapelet_id}/stats/matrix-cell-detail",
    response_model=ShapeletMatrixCellDetailResponse,
)
def get_shapelet_matrix_cell_detail(
    dataset_name: str,
    shapelet_id: str,
    row_index: int = Query(ge=0),
    time_bin_index: int = Query(ge=0),
    scope: str = Query(default="test"),
    omega: float = Query(default=DEFAULT_EXPLAIN_OMEGA),
    time_bins: int = Query(default=180, ge=24, le=400),
    row_bins: int = Query(default=120, ge=24, le=300),
    aggregation: str = Query(default="max"),
    normalization: str = Query(default="none"),
    sort_mode: str = Query(default="peak_position"),
) -> ShapeletMatrixCellDetailResponse:
    return service.get_matrix_cell_detail(
        dataset_name,
        shapelet_id,
        row_index,
        time_bin_index,
        scope,
        omega,
        time_bins,
        row_bins,
        aggregation,
        normalization,
        sort_mode,
    )


@router.get("/datasets/{dataset_name}/shapelets/{shapelet_id}/stats/classes", response_model=ShapeletClassStatsResponse)
def get_shapelet_class_stats(
    dataset_name: str,
    shapelet_id: str,
    scope: str = Query(default="test"),
    omega: float = Query(default=DEFAULT_EXPLAIN_OMEGA),
) -> ShapeletClassStatsResponse:
    return service.get_class_stats(dataset_name, shapelet_id, scope, omega)


@router.get("/datasets/{dataset_name}/shapelets/{shapelet_id}/samples/top-hits", response_model=ShapeletTopHitsResponse)
def get_shapelet_top_hits(
    dataset_name: str,
    shapelet_id: str,
    scope: str = Query(default="test"),
    omega: float = Query(default=DEFAULT_EXPLAIN_OMEGA),
    class_id: int | None = Query(default=None),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
    rank_metric: str = Query(default="max_i"),
) -> ShapeletTopHitsResponse:
    return service.get_top_hits(dataset_name, shapelet_id, scope, omega, class_id, offset, limit, rank_metric)
