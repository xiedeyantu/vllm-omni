# adapted from sglang and fastvideo
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field

from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType


@dataclass
class OmniUniversalRequest:
    prompts: list[OmniPromptType]  # Actually supporting str-based prompts
    request_ids: list[str] = field(default_factory=list)
