#!/usr/bin/env python3
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

import math
import unittest

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.optimizers.slaclipoptimizer import SlaClipDPOptimizer
from torch.utils.data import DataLoader, TensorDataset


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


class SlaClipDPOptimizerTest(unittest.TestCase):
    def _make_optimizer(
        self,
        *,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 2.0,
        expected_batch_size: int = 1,
        num_slots: int = 2,
        eta: float = 0.5,
        beta: float = 0.5,
        c_min: float = 0.1,
        c_max: float = 50.0,
    ):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        dp_optimizer = SlaClipDPOptimizer(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            num_slots=num_slots,
            eta=eta,
            beta=beta,
            c_min=c_min,
            c_max=c_max,
            strict_paper_check=True,
        )
        return model, dp_optimizer

    def test_update_equation_matches_same_query_rule(self):
        _model, optimizer = self._make_optimizer()
        s_hat = torch.tensor([0.2, 0.4])
        clip_value = 2.0

        target_ratio = max(0.0, min(1.0, 1.0 - 0.5 * (1.0 - (0.4 / 2.0))))
        expected = clip_value * math.exp(0.5 * (target_ratio - 0.2))

        actual = optimizer._update_threshold(clip_value, s_hat)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_slack_vector_preserves_extended_norm_bound(self):
        _model, optimizer = self._make_optimizer(max_grad_norm=2.0, num_slots=4)
        clip_value = 2.0
        norms = torch.tensor([0.0, 0.5, 1.25, 2.0, 3.0], dtype=torch.float32)
        slack_amount = torch.clamp(clip_value - norms, min=0.0)
        lambda_t = clip_value / math.sqrt(optimizer.num_slots)
        slack_vector = optimizer._build_slack_vector(
            slack_amount * math.sqrt(optimizer.num_slots), lambda_t
        )
        clipped_norms = torch.clamp(norms, max=clip_value)
        extended_norms = torch.sqrt(
            clipped_norms.square() + slack_vector.square().sum(dim=1)
        )
        self.assertTrue(torch.all(extended_norms <= (clip_value + 1e-6)))

    def test_slack_indicator_uses_expected_batch_size_under_mean_reduction(self):
        model, optimizer = self._make_optimizer(
            noise_multiplier=0.0,
            max_grad_norm=2.0,
            expected_batch_size=8,
            num_slots=2,
        )
        param = next(model.parameters())
        param.grad_sample = torch.zeros((4,) + tuple(param.shape), dtype=param.dtype)
        param.summed_grad = torch.zeros_like(param)

        optimizer._slack_sum = torch.tensor([16.0, 8.0], dtype=torch.float32)
        optimizer._slot_width = 2.0

        optimizer.add_noise()

        expected = torch.tensor([1.0, 0.5], dtype=torch.float32)
        self.assertTrue(torch.allclose(optimizer._slack_indicator, expected))

    def test_slack_sum_accumulates_across_clip_calls(self):
        model, optimizer = self._make_optimizer(
            noise_multiplier=0.0,
            max_grad_norm=2.0,
            expected_batch_size=4,
            num_slots=4,
        )
        param = next(model.parameters())

        param.grad_sample = torch.tensor([[[0.5]], [[0.5]]], dtype=param.dtype)
        optimizer.clip_and_accumulate()

        param.grad_sample = torch.tensor([[[1.0]], [[1.0]]], dtype=param.dtype)
        optimizer.clip_and_accumulate()

        expected = torch.tensor([4.0, 4.0, 2.0, 0.0], dtype=torch.float32)
        self.assertTrue(torch.equal(optimizer._slack_sum, expected))

    def test_privacy_engine_returns_slaclip_optimizer(self):
        torch.manual_seed(0)
        data = torch.randn(16, 10)
        target = torch.randint(0, 2, (16,))
        loader = DataLoader(TensorDataset(data, target), batch_size=8)

        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        privacy_engine = PrivacyEngine()
        model, dp_optimizer, private_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            clipping="slaclip",
            grad_sample_mode="hooks",
            num_slots=8,
            eta=0.5,
            beta=0.5,
            c_min=0.1,
            c_max=50.0,
        )

        self.assertIsInstance(dp_optimizer, SlaClipDPOptimizer)
        self.assertEqual(dp_optimizer.num_slots, 8)
        self.assertEqual(dp_optimizer.eta, 0.5)
        self.assertEqual(dp_optimizer.beta, 0.5)
        self.assertEqual(len(private_loader), len(loader))

    def test_accountant_history_matches_flat_clipping(self):
        torch.manual_seed(0)
        data = torch.randn(32, 10)
        target = torch.randint(0, 2, (32,))
        loader = DataLoader(TensorDataset(data, target), batch_size=8)
        criterion = nn.CrossEntropyLoss()

        def train_one_step(clipping: str):
            model = SimpleModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            privacy_engine = PrivacyEngine(accountant="rdp")
            model, dp_optimizer, private_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=loader,
                noise_multiplier=1.0,
                max_grad_norm=1.0,
                clipping=clipping,
                grad_sample_mode="hooks",
                num_slots=8,
                eta=0.5,
                beta=0.5,
                c_min=0.1,
                c_max=50.0,
            )

            batch_x, batch_y = next(iter(private_loader))
            dp_optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            dp_optimizer.step()
            return privacy_engine.accountant.history

        flat_history = train_one_step("flat")
        slaclip_history = train_one_step("slaclip")
        self.assertEqual(flat_history, slaclip_history)


if __name__ == "__main__":
    unittest.main()
