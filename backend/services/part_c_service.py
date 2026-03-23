from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import HTTPException

from backend.core.constants import DEFAULT_EXPLAIN_OMEGA, SUPPORTED_DATASETS
from backend.schemas.part_a import ApiWarning, PredictionSummary
from backend.schemas.part_c import (
    HighlightWindow,
    MatchParams,
    MatchTensorResponse,
    PartBToCLink,
    PartCFromPartEResponse,
    PartCFromPartBResponse,
    PartCMatchRequest,
    PartCMetaResponse,
    PartEToCLink,
    PinnedShapeletStatus,
)
from backend.services.part_a_service import PartAService, _to_numpy_2d

_SCOPE_VALUES = {"test", "train"}
_SHAPELET_ID_RE = re.compile(r"\d+")
_MAX_SHAPELETS_REQUESTED = 512


def _http_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


def _normalize_softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def _margin_from_probs(probs: np.ndarray) -> float:
    if probs.size <= 1:
        return 1.0
    top2 = np.sort(probs)[-2:]
    return float(top2[-1] - top2[-2])


class PartCService:
    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2])
        self.part_a_service = PartAService(self.project_root)
        self.inference_device = self.part_a_service.inference_device
        self.scope_default = "test"
        self.omega_default = float(DEFAULT_EXPLAIN_OMEGA)

    def _build_navigation_match_request(
        self,
        scope: str,
        omega: float,
        pinned_shapelet_id: str,
        include_sequence: bool,
        include_prediction: bool,
        include_windows: bool,
    ) -> PartCMatchRequest:
        return PartCMatchRequest(
            scope=scope,
            omega=omega,
            shapelet_ids=None,
            topk_shapelets=None,
            pinned_shapelet_id=pinned_shapelet_id,
            include_sequence=include_sequence,
            include_prediction=include_prediction,
            include_windows=include_windows,
        )

    def _validate_scope(self, scope: str) -> str:
        normalized = (scope or "").lower().strip()
        if normalized not in _SCOPE_VALUES:
            raise _http_error(400, "ERR_INVALID_SCOPE", "scope must be one of: test, train")
        return normalized

    def _validate_omega(self, omega: float) -> float:
        if not math.isfinite(omega):
            raise _http_error(400, "ERR_INVALID_OMEGA", "omega must be a finite float")
        return float(omega)

    def _validate_topk(self, topk_shapelets: int | None) -> int | None:
        if topk_shapelets is None:
            return None
        if topk_shapelets < 1 or topk_shapelets > _MAX_SHAPELETS_REQUESTED:
            raise _http_error(
                400,
                "ERR_INVALID_TOPK",
                f"topk_shapelets must be in [1, {_MAX_SHAPELETS_REQUESTED}]",
            )
        return int(topk_shapelets)

    def _parse_sample_id(self, sample_id: str) -> int:
        try:
            return int(sample_id)
        except ValueError as exc:
            raise _http_error(404, "ERR_SAMPLE_NOT_FOUND", f"Invalid sample_id: {sample_id}") from exc

    def _parse_shapelet_index(self, shapelet_id: str, num_shapelets: int) -> int:
        matches = _SHAPELET_ID_RE.findall(shapelet_id)
        if not matches:
            raise _http_error(404, "ERR_SHAPELET_NOT_FOUND", f"Invalid shapelet_id: {shapelet_id}")
        idx = int(matches[-1])
        if idx < 0 or idx >= num_shapelets:
            raise _http_error(404, "ERR_SHAPELET_NOT_FOUND", f"shapelet_id out of range: {shapelet_id}")
        return idx

    @lru_cache(maxsize=len(SUPPORTED_DATASETS))
    def _load_seg_model(self, dataset_name: str) -> Any:
        dataset_name = dataset_name.lower()
        if dataset_name not in SUPPORTED_DATASETS:
            raise _http_error(404, "ERR_DATASET_NOT_FOUND", f"Unsupported dataset: {dataset_name}")

        from omegaconf import OmegaConf

        from exp_saliency_general import ShapeXPipline

        config = OmegaConf.create(
            {
                "base": {"root_dir": str(self.project_root)},
                "dataset": {"name": dataset_name, "meta_dataset": "default"},
            }
        )
        pipeline = ShapeXPipline(config)
        pipeline.args.root_path = str(self.project_root)
        pipeline.args.use_gpu = self.inference_device.startswith("cuda")
        pipeline.args.device = self.inference_device
        seg_model = pipeline.load_seg_model().to(self.inference_device)
        seg_model.eval()
        return seg_model

    def _predict_activations_single(self, seg_model: Any, sequence: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            seq = sequence.to(next(seg_model.parameters()).device)
            _, activations, _ = seg_model(seq, seq, seq, seq)
        return activations.detach().cpu().numpy()[0]  # [T, P]

    def _resolve_dataset_and_sample(self, dataset_name: str, scope: str, sample_id: str) -> tuple[Any, int, Any]:
        bundle = self.part_a_service.load_dataset_bundle(dataset_name)
        dataset = bundle.test_dataset if scope == "test" else bundle.train_dataset
        sample_idx = self._parse_sample_id(sample_id)
        labels = self.part_a_service._dataset_labels(dataset)
        if sample_idx < 0 or sample_idx >= int(labels.shape[0]):
            raise _http_error(404, "ERR_SAMPLE_NOT_FOUND", f"sample_id out of range: {sample_id}")
        return bundle, sample_idx, dataset

    def _shapelet_ids_for_size(self, num_shapelets: int) -> list[str]:
        return [f"s{idx:04d}" for idx in range(num_shapelets)]

    def _resolve_selected_indices(
        self,
        request: PartCMatchRequest,
        max_scores: np.ndarray,
        all_shapelet_ids: list[str],
        warnings: list[ApiWarning],
    ) -> list[int]:
        num_shapelets = len(all_shapelet_ids)
        topk = self._validate_topk(request.topk_shapelets)

        if request.shapelet_ids is not None and len(request.shapelet_ids) > _MAX_SHAPELETS_REQUESTED:
            raise _http_error(
                400,
                "ERR_TOO_MANY_SHAPELETS",
                f"shapelet_ids length exceeds {_MAX_SHAPELETS_REQUESTED}",
            )

        if request.shapelet_ids:
            deduped: list[str] = []
            seen = set()
            for sid in request.shapelet_ids:
                if sid in seen:
                    continue
                seen.add(sid)
                deduped.append(sid)
            if request.topk_shapelets is not None:
                warnings.append(
                    ApiWarning(
                        code="TOPK_IGNORED_BY_SHAPELET_IDS",
                        message="shapelet_ids is provided; topk_shapelets is ignored.",
                    )
                )
            return [self._parse_shapelet_index(sid, num_shapelets) for sid in deduped]

        if topk is not None:
            order = sorted(range(num_shapelets), key=lambda i: (-float(max_scores[i]), all_shapelet_ids[i]))
            return order[:topk]

        return list(range(num_shapelets))

    def _build_windows(
        self,
        selected_indices: list[int],
        all_shapelet_ids: list[str],
        shapelet_lens_all: list[int],
        peak_t_all: list[int],
        max_scores_all: list[float],
        omega: float,
        seq_len: int,
    ) -> list[HighlightWindow]:
        windows: list[HighlightWindow] = []
        for idx in selected_indices:
            peak_t = int(peak_t_all[idx])
            peak_score = float(max_scores_all[idx])
            triggered = peak_score >= omega
            if not triggered:
                continue
            shapelet_len = int(shapelet_lens_all[idx])
            start = peak_t - (shapelet_len // 2)
            end = start + shapelet_len - 1
            start = max(0, start)
            end = min(seq_len - 1, end)
            windows.append(
                HighlightWindow(
                    shapelet_id=all_shapelet_ids[idx],
                    shapelet_len=shapelet_len,
                    peak_t=peak_t,
                    start=start,
                    end=end,
                    peak_score=peak_score,
                    triggered=triggered,
                )
            )
        return windows

    def _pinned_status(
        self,
        pinned_shapelet_id: str | None,
        num_shapelets: int,
        selected_indices: list[int],
        all_shapelet_ids: list[str],
        peak_t_all: list[int],
        max_scores_all: list[float],
        omega: float,
        warnings: list[ApiWarning],
    ) -> PinnedShapeletStatus:
        if pinned_shapelet_id is None:
            return PinnedShapeletStatus(shapelet_id=None, is_present_in_tensor=False, peak_t=None, triggered=None)

        pinned_idx = self._parse_shapelet_index(pinned_shapelet_id, num_shapelets)
        is_present = pinned_idx in selected_indices
        if not is_present:
            warnings.append(
                ApiWarning(
                    code="PINNED_SHAPELET_NOT_IN_RESPONSE",
                    message=(
                        "Pinned shapelet is valid but not present in the returned tensor. "
                        "This can happen when requesting a subset."
                    ),
                )
            )
        return PinnedShapeletStatus(
            shapelet_id=all_shapelet_ids[pinned_idx],
            is_present_in_tensor=is_present,
            peak_t=int(peak_t_all[pinned_idx]),
            triggered=bool(float(max_scores_all[pinned_idx]) >= omega),
        )

    def _prediction_summary(self, bundle: Any, sequence: torch.Tensor) -> PredictionSummary:
        logits = self.part_a_service._predict_batch(bundle.class_model, sequence)
        probs = _normalize_softmax(logits)[0]
        return PredictionSummary(
            pred_class=int(np.argmax(probs)),
            probs=[float(v) for v in probs.tolist()],
            margin=_margin_from_probs(probs),
        )

    def get_meta(self, dataset_name: str) -> PartCMetaResponse:
        bundle = self.part_a_service.load_dataset_bundle(dataset_name)
        return PartCMetaResponse(
            dataset=bundle.dataset_name,
            scope_default=self.scope_default,
            omega_default=self.omega_default,
            warnings=bundle.warnings,
        )

    def get_match_tensor(
        self,
        dataset_name: str,
        sample_id: str,
        request: PartCMatchRequest,
    ) -> MatchTensorResponse:
        scope = self._validate_scope(request.scope)
        omega = self._validate_omega(request.omega)
        bundle, sample_idx, dataset = self._resolve_dataset_and_sample(dataset_name, scope, sample_id)
        warnings = list(bundle.warnings)

        sequence = self.part_a_service._dataset_sequence_at(dataset, sample_idx).unsqueeze(0)
        seq_len = int(sequence.shape[1])
        seg_model = self._load_seg_model(dataset_name)

        activations_tp = self._predict_activations_single(seg_model, sequence)  # [T, P]
        activations_pt = activations_tp.transpose(1, 0)  # [P, T]
        max_scores = activations_pt.max(axis=1)
        peak_t = activations_pt.argmax(axis=1)

        prototype_vectors = seg_model.prototype_vectors.detach().cpu().numpy()
        num_shapelets = int(prototype_vectors.shape[0])
        shapelet_len = int(prototype_vectors.shape[1])
        all_shapelet_ids = self._shapelet_ids_for_size(num_shapelets)
        all_shapelet_lens = [shapelet_len] * num_shapelets
        all_peak_t = [int(v) for v in peak_t.tolist()]
        all_max_scores = [float(v) for v in max_scores.tolist()]

        selected_indices = self._resolve_selected_indices(request, max_scores, all_shapelet_ids, warnings)

        selected_shapelet_ids = [all_shapelet_ids[idx] for idx in selected_indices]
        selected_shapelet_lens = [all_shapelet_lens[idx] for idx in selected_indices]
        selected_peak_t = [all_peak_t[idx] for idx in selected_indices]
        selected_i = [[float(v) for v in activations_pt[idx].tolist()] for idx in selected_indices]

        windows = None
        if request.include_windows:
            windows = self._build_windows(
                selected_indices,
                all_shapelet_ids,
                all_shapelet_lens,
                all_peak_t,
                all_max_scores,
                omega,
                seq_len,
            )

        pinned = self._pinned_status(
            request.pinned_shapelet_id,
            num_shapelets,
            selected_indices,
            all_shapelet_ids,
            all_peak_t,
            all_max_scores,
            omega,
            warnings,
        )

        response_sequence = _to_numpy_2d(sequence[0]).tolist() if request.include_sequence else None
        response_prediction = self._prediction_summary(bundle, sequence) if request.include_prediction else None

        normalize = "znorm" if bool(getattr(bundle.args, "shapelet_znorm", False)) else "none"
        params = MatchParams(
            similarity_type=str(getattr(bundle.args, "dist_measure", "unknown")),
            shapelet_temperature=float(getattr(bundle.args, "shapelet_temperature", 1.0)),
            normalize=normalize,
        )

        return MatchTensorResponse(
            dataset=bundle.dataset_name,
            sample_id=str(sample_idx),
            split=scope,
            scope=scope,
            omega=omega,
            shapelet_ids=selected_shapelet_ids,
            shapelet_lens=selected_shapelet_lens,
            I=selected_i,
            peak_t=selected_peak_t,
            windows=windows,
            pinned_shapelet=pinned,
            sequence=response_sequence,
            prediction=response_prediction,
            params=params,
            warnings=warnings,
        )

    def from_part_b_navigation(
        self,
        request_link: PartBToCLink,
        include_sequence: bool,
        include_prediction: bool,
        include_windows: bool,
    ) -> PartCFromPartBResponse:
        if request_link.source_panel != "part_b":
            raise _http_error(400, "ERR_INVALID_SOURCE_PANEL", "source_panel must be 'part_b'")
        resolved_request = self._build_navigation_match_request(
            scope=request_link.scope,
            omega=request_link.omega,
            pinned_shapelet_id=request_link.shapelet_id,
            include_sequence=include_sequence,
            include_prediction=include_prediction,
            include_windows=include_windows,
        )
        match = self.get_match_tensor(request_link.dataset, request_link.sample_id, resolved_request)
        warnings = list(match.warnings)
        return PartCFromPartBResponse(
            link=request_link,
            resolved_match_request=resolved_request,
            match=match,
            warnings=warnings,
        )

    def from_part_e_navigation(
        self,
        request_link: PartEToCLink,
        include_sequence: bool,
        include_prediction: bool,
        include_windows: bool,
    ) -> PartCFromPartEResponse:
        if request_link.source_panel != "part_e":
            raise _http_error(400, "ERR_INVALID_SOURCE_PANEL", "source_panel must be 'part_e'")
        if request_link.t_start < 0 or request_link.t_end < 0 or request_link.t_start > request_link.t_end:
            raise _http_error(400, "ERR_INVALID_SPAN", "span must satisfy 0 <= t_start <= t_end")

        resolved_request = self._build_navigation_match_request(
            scope=request_link.scope,
            omega=request_link.omega,
            pinned_shapelet_id=request_link.shapelet_id,
            include_sequence=include_sequence,
            include_prediction=include_prediction,
            include_windows=include_windows,
        )
        match = self.get_match_tensor(request_link.dataset, request_link.sample_id, resolved_request)
        warnings = list(match.warnings)
        return PartCFromPartEResponse(
            link=request_link,
            resolved_match_request=resolved_request,
            match=match,
            warnings=warnings,
        )
