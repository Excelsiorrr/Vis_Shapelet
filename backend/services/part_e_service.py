from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import HTTPException

from backend.core.constants import SUPPORTED_DATASETS
from backend.schemas.part_a import ApiWarning
from backend.schemas.part_e import PartEWhatIfRequest, PartEWhatIfResponse
from backend.services.part_a_service import PartAService, _to_numpy_2d

_SCOPE_VALUES = {"test", "train"}
_BASELINE_VALUES = {"linear_interp", "zero", "dataset_mean"}
_VALUE_TYPE_VALUES = {"prob", "logit"}
_SHAPELET_ID_RE = re.compile(r"\d+")
_DEFAULT_SEED = 2026


def _http_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


def _normalize_softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


class PartEService:
    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2])
        self.part_a_service = PartAService(self.project_root)
        self.inference_device = self.part_a_service.inference_device

    def _validate_scope(self, scope: str) -> str:
        normalized = (scope or "").lower().strip()
        if normalized not in _SCOPE_VALUES:
            raise _http_error(400, "ERR_INVALID_SCOPE", "scope must be one of: test, train")
        return normalized

    def _validate_omega(self, omega: float) -> float:
        if not math.isfinite(omega):
            raise _http_error(400, "ERR_INVALID_OMEGA", "omega must be a finite float")
        return float(omega)

    def _validate_baseline(self, baseline: str) -> str:
        normalized = (baseline or "").lower().strip()
        if normalized not in _BASELINE_VALUES:
            raise _http_error(
                400,
                "ERR_INVALID_BASELINE",
                f"baseline must be one of: {sorted(_BASELINE_VALUES)}",
            )
        return normalized

    def _validate_value_type(self, value_type: str) -> str:
        normalized = (value_type or "").lower().strip()
        if normalized not in _VALUE_TYPE_VALUES:
            raise _http_error(
                400,
                "ERR_INVALID_VALUE_TYPE",
                f"value_type must be one of: {sorted(_VALUE_TYPE_VALUES)}",
            )
        return normalized

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

    def _resolve_dataset_and_sample(
        self,
        dataset_name: str,
        scope: str,
        sample_id: str,
    ) -> tuple[Any, Any, int]:
        bundle = self.part_a_service.load_dataset_bundle(dataset_name)
        dataset = bundle.test_dataset if scope == "test" else bundle.train_dataset
        sample_idx = self._parse_sample_id(sample_id)
        labels = self.part_a_service._dataset_labels(dataset)
        if sample_idx < 0 or sample_idx >= int(labels.shape[0]):
            raise _http_error(404, "ERR_SAMPLE_NOT_FOUND", f"sample_id out of range: {sample_id}")
        return bundle, dataset, sample_idx

    def _predict_logits(self, class_model: Any, sequence: torch.Tensor) -> np.ndarray:
        return self.part_a_service._predict_batch(class_model, sequence)[0]

    def _clip_span(self, t_start: int, t_end: int, seq_len: int, warnings: list[ApiWarning]) -> tuple[int, int]:
        if t_start > t_end:
            raise _http_error(400, "ERR_INVALID_SPAN", "t_start must be <= t_end")
        clipped_start = max(0, min(t_start, seq_len - 1))
        clipped_end = max(0, min(t_end, seq_len - 1))
        if clipped_start > clipped_end:
            raise _http_error(400, "ERR_INVALID_SPAN", "span becomes invalid after clipping")
        if clipped_start != t_start or clipped_end != t_end:
            warnings.append(
                ApiWarning(
                    code="SPAN_CLIPPED",
                    message=(
                        f"Requested span [{t_start},{t_end}] is clipped to "
                        f"[{clipped_start},{clipped_end}] within [0,{seq_len - 1}]."
                    ),
                )
            )
        return clipped_start, clipped_end

    def _apply_linear_interp(self, sequence: np.ndarray, t_start: int, t_end: int) -> np.ndarray:
        perturbed = sequence.copy()
        seq_len, channels = perturbed.shape
        left_idx = t_start - 1 if t_start - 1 >= 0 else None
        right_idx = t_end + 1 if t_end + 1 < seq_len else None
        seg_len = t_end - t_start + 1

        for dim in range(channels):
            left_val = perturbed[left_idx, dim] if left_idx is not None else None
            right_val = perturbed[right_idx, dim] if right_idx is not None else None
            if left_val is not None and right_val is not None:
                for k, t in enumerate(range(t_start, t_end + 1), start=1):
                    alpha = k / (seg_len + 1)
                    perturbed[t, dim] = (1 - alpha) * left_val + alpha * right_val
            elif left_val is not None:
                perturbed[t_start : t_end + 1, dim] = left_val
            elif right_val is not None:
                perturbed[t_start : t_end + 1, dim] = right_val
            else:
                # T==1 and the span is the entire series: keep original value.
                perturbed[t_start : t_end + 1, dim] = perturbed[t_start : t_end + 1, dim]
        return perturbed

    def _apply_baseline(
        self,
        baseline: str,
        sequence: np.ndarray,
        t_start: int,
        t_end: int,
        dataset_sequences: torch.Tensor,
    ) -> np.ndarray:
        perturbed = sequence.copy()
        if baseline == "zero":
            perturbed[t_start : t_end + 1, :] = 0.0
            return perturbed
        if baseline == "dataset_mean":
            dataset_mean = dataset_sequences.float().mean(dim=(0, 1)).detach().cpu().numpy()
            perturbed[t_start : t_end + 1, :] = dataset_mean
            return perturbed
        return self._apply_linear_interp(perturbed, t_start, t_end)

    def _shapelet_span_mismatch_warning(
        self,
        dataset_name: str,
        shapelet_id: str,
        sequence: torch.Tensor,
        t_start: int,
        t_end: int,
        warnings: list[ApiWarning],
    ) -> None:
        seg_model = self._load_seg_model(dataset_name)
        with torch.no_grad():
            seq = sequence.to(next(seg_model.parameters()).device)
            _, activations, _ = seg_model(seq, seq, seq, seq)
        activations_tp = activations.detach().cpu().numpy()[0]  # [T, P]
        num_shapelets = int(activations_tp.shape[1])
        shapelet_idx = self._parse_shapelet_index(shapelet_id, num_shapelets)
        peak_t = int(np.argmax(activations_tp[:, shapelet_idx]))

        if not (t_start <= peak_t <= t_end):
            warnings.append(
                ApiWarning(
                    code="SHAPELET_SPAN_MISMATCH",
                    message=(
                        f"shapelet_id={shapelet_id} peak_t={peak_t} is outside requested span "
                        f"[{t_start},{t_end}]. Evaluation continues."
                    ),
                )
            )

    def evaluate_whatif(
        self,
        dataset_name: str,
        sample_id: str,
        request: PartEWhatIfRequest,
    ) -> PartEWhatIfResponse:
        scope = self._validate_scope(request.scope)
        omega = self._validate_omega(request.omega)
        baseline = self._validate_baseline(request.baseline)
        value_type = self._validate_value_type(request.value_type)
        seed = int(request.seed if request.seed is not None else _DEFAULT_SEED)

        bundle, dataset, sample_idx = self._resolve_dataset_and_sample(dataset_name, scope, sample_id)
        warnings = list(bundle.warnings)

        sequence = self.part_a_service._dataset_sequence_at(dataset, sample_idx)
        sequence_np = _to_numpy_2d(sequence)
        seq_len = int(sequence_np.shape[0])
        t_start, t_end = self._clip_span(request.t_start, request.t_end, seq_len, warnings)

        sample_batch = sequence.unsqueeze(0)
        self._shapelet_span_mismatch_warning(
            dataset_name=bundle.dataset_name,
            shapelet_id=request.shapelet_id,
            sequence=sample_batch,
            t_start=t_start,
            t_end=t_end,
            warnings=warnings,
        )

        dataset_sequences = self.part_a_service._dataset_sequences(dataset)
        perturbed_np = self._apply_baseline(
            baseline=baseline,
            sequence=sequence_np,
            t_start=t_start,
            t_end=t_end,
            dataset_sequences=dataset_sequences,
        )

        perturbed_tensor = torch.tensor(perturbed_np, dtype=torch.float32).unsqueeze(0)

        logits_original = self._predict_logits(bundle.class_model, sample_batch)
        logits_whatif = self._predict_logits(bundle.class_model, perturbed_tensor)

        probs_original = _normalize_softmax(logits_original[np.newaxis, :])[0]
        probs_whatif = _normalize_softmax(logits_whatif[np.newaxis, :])[0]
        pred_class_original = int(np.argmax(probs_original))
        pred_class_whatif = int(np.argmax(probs_whatif))

        vec_original = probs_original if value_type == "prob" else logits_original
        vec_whatif = probs_whatif if value_type == "prob" else logits_whatif

        p_original = float(vec_original[pred_class_original])
        p_whatif = float(vec_whatif[pred_class_original])
        delta = float(p_whatif - p_original)

        delta_target: float | None = None
        if request.target_class is not None:
            target_class = int(request.target_class)
            if 0 <= target_class < int(vec_original.shape[0]):
                delta_target = float(vec_whatif[target_class] - vec_original[target_class])
            else:
                warnings.append(
                    ApiWarning(
                        code="TARGET_CLASS_OUT_OF_RANGE",
                        message=(
                            f"target_class={target_class} is outside valid range "
                            f"[0,{int(vec_original.shape[0]) - 1}]. delta_target is null."
                        ),
                    )
                )

        labels = self.part_a_service._dataset_labels(dataset)
        y_true = int(labels[sample_idx].item()) if sample_idx < int(labels.shape[0]) else None

        perturbed_sequence = perturbed_np.tolist() if request.include_perturbed_sequence else None

        return PartEWhatIfResponse(
            dataset=bundle.dataset_name,
            sample_id=str(sample_idx),
            shapelet_id=request.shapelet_id,
            t_start=t_start,
            t_end=t_end,
            scope=scope,
            omega=omega,
            baseline=baseline,
            value_type=value_type,
            target_class=request.target_class,
            seed=seed,
            p_original=p_original,
            p_whatif=p_whatif,
            delta=delta,
            delta_target=delta_target,
            pred_class_original=pred_class_original,
            pred_class_whatif=pred_class_whatif,
            y_true=y_true,
            perturbed_sequence=perturbed_sequence,
            warnings=warnings,
        )
