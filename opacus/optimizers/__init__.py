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

from .adaclipoptimizer import AdaClipDPOptimizer
from .ddp_perlayeroptimizer import SimpleDistributedPerLayerOptimizer
from .ddpoptimizer import DistributedDPOptimizer
from .ddpoptimizer_fast_gradient_clipping import (
    DistributedDPOptimizerFastGradientClipping,
)
from .fsdpoptimizer_fast_gradient_clipping import FSDPOptimizerFastGradientClipping
from .optimizer import DPOptimizer
from .optimizer_fast_gradient_clipping import DPOptimizerFastGradientClipping
from .perlayeroptimizer import DPPerLayerOptimizer
from .slaclipoptimizer import SlaClipDPOptimizer


__all__ = [
    "AdaClipDPOptimizer",
    "DistributedDPOptimizer",
    "DPOptimizer",
    "DPOptimizerFastGradientClipping",
    "DistributedDPOptimizerFastGradientlipping",
    "FSDPOptimizerFastGradientClipping",
    "DPPerLayerOptimizer",
    "SlaClipDPOptimizer",
    "SimpleDistributedPerLayerOptimizer",
]


def get_optimizer_class(clipping: str, distributed: bool, grad_sample_mode: str = None):
    if grad_sample_mode == "ghost":
        if clipping != "flat":
            raise ValueError(
                f"Unsupported combination of parameters. Clipping: {clipping} and grad_sample_mode: {grad_sample_mode}"
            )
        if distributed is False:
            return DPOptimizerFastGradientClipping
        return DistributedDPOptimizerFastGradientClipping
    elif grad_sample_mode == "ghost_fsdp":
        if clipping != "flat" or distributed is not True:
            raise ValueError(
                f"Unsupported combination of parameters. Clipping: {clipping}, distributed: {distributed}, and grad_sample_mode: {grad_sample_mode}"
            )
        return FSDPOptimizerFastGradientClipping

    if clipping == "flat":
        return DistributedDPOptimizer if distributed else DPOptimizer

    if clipping == "per_layer":
        if distributed is False:
            return DPPerLayerOptimizer
        if grad_sample_mode not in ("hooks", "ew"):
            raise ValueError(f"Unexpected grad_sample_mode: {grad_sample_mode}")
        return SimpleDistributedPerLayerOptimizer

    if clipping == "adaptive" and distributed is False:
        return AdaClipDPOptimizer

    if clipping == "slaclip" and distributed is False:
        return SlaClipDPOptimizer

    raise ValueError(
        f"Unexpected optimizer parameters. Clipping: {clipping}, distributed: {distributed}"
    )
