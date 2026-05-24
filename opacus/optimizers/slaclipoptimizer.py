# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.optim import Optimizer

from .optimizer import (
    DPOptimizer,
    _check_processed_flag,
    _generate_noise,
    _mark_as_processed,
)


class SlaClipDPOptimizer(DPOptimizer):
    """
    :class:`~opacus.optimizers.optimizer.DPOptimizer` variant implementing
    SlaClip adaptive clipping with a same-query Gaussian release.

    The first ``d`` released coordinates coincide with vanilla clipped gradients,
    while the extra ``K`` coordinates encode clipping slacks. Both parts share the
    same Gaussian release, so clipping adaptation does not introduce an extra
    privacy query relative to vanilla DP-SGD under the same sampling rule and
    noise multiplier.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        num_slots: int,
        eta: float,
        beta: float = 0.5,
        c_min: float = 0.1,
        c_max: float = 50.0,
        strict_paper_check: bool = True,
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        **kwargs,
    ):
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
        )

        self.num_slots = int(num_slots)
        if self.num_slots <= 0:
            raise ValueError("num_slots must be a positive integer")
        if c_max <= c_min:
            raise ValueError("c_max must be larger than c_min")

        self.eta = float(eta)
        self.beta = float(beta)
        self.c_min = float(c_min)
        self.c_max = float(c_max)
        self.strict_paper_check = bool(strict_paper_check)
        self._stability_eps = 1e-6

        self._slack_sum: Optional[torch.Tensor] = None
        self._slot_width: float = 0.0
        self._slack_indicator: Optional[torch.Tensor] = None

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients and SlaClip-specific state.

        When the last step is skipped (for example under virtual batching),
        ``p.summed_grad`` should be preserved, matching ``DPOptimizer`` behavior.
        In that case the slack state is also preserved so the final joint release
        is computed over all accumulated physical batches.
        """
        super().zero_grad(set_to_none)
        self._slack_indicator = None
        if not self._is_last_step_skipped:
            self._slack_sum = None
            self._slot_width = 0.0

    def _release_denom(self) -> float:
        if self.loss_reduction == "sum":
            return 1.0

        denom = float(self.expected_batch_size) * float(self.accumulated_iterations)
        if denom <= 0:
            raise ValueError("Expected release denominator must be > 0")
        return denom

    def _build_slack_vector(
        self, slack_extent: torch.Tensor, slot_width: float
    ) -> torch.Tensor:
        """
        Encode per-sample clipping slack into ``num_slots`` coordinates.

        The resulting vector has the same norm contribution as the paper's
        slack encoding while remaining convenient for a joint Gaussian release.
        """
        batch_size = int(slack_extent.shape[0])
        slack_vector = torch.zeros(
            batch_size,
            int(self.num_slots),
            device=slack_extent.device,
            dtype=torch.float32,
        )
        if slot_width <= 0:
            return slack_vector

        quotient = torch.floor(slack_extent / slot_width).to(torch.int64)
        quotient_clamped = torch.clamp(quotient, max=int(self.num_slots))
        residual = slack_extent - quotient_clamped.to(slack_extent.dtype) * slot_width
        residual = torch.where(
            quotient_clamped >= int(self.num_slots),
            torch.zeros_like(residual),
            residual,
        )

        arange_k = torch.arange(int(self.num_slots), device=slack_extent.device).view(
            1, int(self.num_slots)
        )
        mask = arange_k < quotient_clamped.view(batch_size, 1)
        slack_vector = mask.to(slack_vector.dtype) * float(slot_width)

        valid = quotient_clamped < int(self.num_slots)
        if valid.any():
            indices = torch.clamp(quotient_clamped, max=int(self.num_slots) - 1)
            slack_vector[valid, indices[valid]] = residual[valid]

        return slack_vector

    def clip_and_accumulate(self):
        grad_samples = self.grad_samples
        if grad_samples is None or len(grad_samples) == 0:
            return

        batch_size = None
        device = None
        sum_sq = None
        flat_cache = []

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            flat = self._get_flat_grad_sample(p).to(dtype=torch.float32)
            flat_cache.append((p, flat))

            if batch_size is None:
                batch_size = int(flat.shape[0])
                device = flat.device
                sum_sq = torch.zeros(batch_size, device=device, dtype=torch.float32)
            elif int(flat.shape[0]) != int(batch_size):
                raise ValueError("Inconsistent batch dimension across parameters")

            sum_sq = sum_sq + (flat.view(batch_size, -1) ** 2).sum(dim=1)

        physical_batch_size = int(batch_size) if batch_size is not None else 0
        if physical_batch_size == 0:
            for p, _flat in flat_cache:
                _mark_as_processed(p.grad_sample)
            return

        clip_bound = float(self.max_grad_norm)
        per_sample_norms = torch.sqrt(sum_sq + 1e-12)
        clip_factor = (clip_bound / (per_sample_norms + 1e-12)).clamp(max=1.0)

        for p, flat in flat_cache:
            grad = torch.einsum("i,i...", clip_factor.to(flat.dtype), flat)
            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad
            _mark_as_processed(p.grad_sample)

        slot_width = float(clip_bound / math.sqrt(self.num_slots))
        if self._slot_width and not math.isclose(
            self._slot_width, slot_width, rel_tol=1e-12, abs_tol=0.0
        ):
            raise ValueError(
                "Inconsistent slot width across accumulated physical batches"
            )
        self._slot_width = slot_width

        slack_amount = torch.clamp(clip_bound - per_sample_norms, min=0.0)
        slack_extent = slack_amount * math.sqrt(self.num_slots)
        slack_vector = self._build_slack_vector(slack_extent, self._slot_width)
        batch_slack_sum = slack_vector.sum(dim=0).to(torch.float32)
        if self._slack_sum is None:
            self._slack_sum = batch_slack_sum
        else:
            if self._slack_sum.device != batch_slack_sum.device:
                batch_slack_sum = batch_slack_sum.to(self._slack_sum.device)
            self._slack_sum += batch_slack_sum

    def _compute_target_ratio(self, clip_bound: float, s_hat: torch.Tensor) -> float:
        """
        Compute the target unclipped ratio used for the next clipping bound.
        """
        if self.strict_paper_check and s_hat.numel() != int(self.num_slots):
            raise ValueError("strict_paper_check: slack_indicator length != num_slots")

        near_zero_mass = float(s_hat[int(self.num_slots) - 1].item())
        normalized_near_zero_mass = near_zero_mass / (clip_bound + self._stability_eps)
        target_ratio = 1.0 - self.beta * (1.0 - normalized_near_zero_mass)
        return float(max(0.0, min(1.0, target_ratio)))

    def _update_threshold(self, clip_bound: float, s_hat: torch.Tensor) -> float:
        noisy_near_threshold_mass = float(s_hat[0].item())
        target_ratio = self._compute_target_ratio(clip_bound, s_hat)
        step = self.eta * (target_ratio - noisy_near_threshold_mass)
        next_clip = float(clip_bound * math.exp(step))
        return float(max(self.c_min, min(self.c_max, next_clip)))

    def add_noise(self):
        clip_bound = float(self.max_grad_norm)

        param_slices = []
        total_dim = 0
        first_dev = None
        for p in self.params:
            if p.summed_grad is None:
                continue
            _check_processed_flag(p.summed_grad)
            if first_dev is None:
                first_dev = p.summed_grad.device
            size = int(p.summed_grad.numel())
            param_slices.append((p, total_dim, size))
            total_dim += size

        if first_dev is None:
            return

        # Draw one joint Gaussian vector and split it into parameter and slack
        # coordinates. This preserves the same-query release used by SlaClip.
        noise_ref = torch.empty(
            total_dim + int(self.num_slots), device=first_dev, dtype=torch.float32
        )
        noise_full = _generate_noise(
            std=self.noise_multiplier * clip_bound,
            reference=noise_ref,
            generator=self.generator,
            secure_mode=self.secure_mode,
        ).to(first_dev, dtype=torch.float32)

        for p, start, size in param_slices:
            ng = noise_full[start : start + size].view_as(p.summed_grad)
            p.grad = (p.summed_grad + ng).view_as(p)
            _mark_as_processed(p.summed_grad)

        if self._slack_sum is None or self._slot_width <= 0:
            return

        slack_sum = self._slack_sum
        if slack_sum.device != first_dev:
            slack_sum = slack_sum.to(first_dev)
        slot_noise = noise_full[total_dim : total_dim + int(self.num_slots)]
        slack_noisy_sum = slack_sum + slot_noise
        s_hat = slack_noisy_sum / (self._slot_width * self._release_denom())
        self._slack_indicator = s_hat

        next_clip = self._update_threshold(clip_bound, s_hat)
        if math.isfinite(next_clip) and next_clip > 0:
            self.max_grad_norm = float(next_clip)
