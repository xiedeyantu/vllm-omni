"""
Engine components for vLLM-Omni.
"""

from typing import Any

import msgspec
import torch
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
)


class PromptEmbedsPayload(msgspec.Struct):
    """Serialized prompt embeddings payload for direct transfer.

    data: raw bytes of the tensor in row-major order
    shape: [seq_len, hidden_size]
    dtype: torch dtype name (e.g., "float16", "float32")
    """

    data: bytes
    shape: list[int]
    dtype: str


class AdditionalInformationEntry(msgspec.Struct):
    """One entry of additional_information.

    Two supported forms are encoded:
      - tensor: data/shape/dtype
      - list: a Python list (msgspec-serializable)
    Exactly one of (tensor_data, list_data) should be non-None.
    """

    # Tensor form
    tensor_data: bytes | None = None
    tensor_shape: list[int] | None = None
    tensor_dtype: str | None = None

    # List form
    list_data: list[Any] | None = None


class AdditionalInformationPayload(msgspec.Struct):
    """Serialized dictionary payload for additional_information.

    Keys are strings; values are encoded as AdditionalInformationEntry.
    """

    entries: dict[str, AdditionalInformationEntry]


class OmniEngineCoreRequest(EngineCoreRequest):
    """Engine core request for omni models with embeddings support.

    Extends the base EngineCoreRequest with support for prompt embeddings
    and additional information payloads, enabling direct transfer of
    pre-computed embeddings between pipeline stages.

    Attributes:
        prompt_embeds: Optional serialized prompt embeddings payload for
            direct transfer between stages
        additional_information: Optional serialized additional information
            dictionary containing tensors or lists to pass along with the request
    """

    # Optional prompt embeddings (direct-transfer version)
    prompt_embeds: PromptEmbedsPayload | None = None
    # Optional additional information dictionary (serialized)
    additional_information: AdditionalInformationPayload | None = None


class OmniEngineCoreOutput(EngineCoreOutput):
    """Engine core output for omni models with multimodal support.
    
    Attributes:
        pooling_output: Optional dictionary of pooling outputs (tensors)
        multimodal_outputs: Dictionary of multimodal outputs from operators
        us_output: Dictionary storing outputs from universal stage operators
            Keys are operator class names (e.g., "VideoLaplacianPipeline")
            Values are operator-specific output dicts containing:
                - "result": The computed result(s)
                - "status": Per-item processing status ("success" / "error")
                - "passed": Per-item pass/fail status
                - Other operator-specific fields
    """
    pooling_output: dict[str, torch.Tensor] | None = None
    multimodal_outputs: dict[str, Any] = {}
    us_output: dict[str, Any] = {}  # Universal stage operator outputs


class OmniEngineCoreOutputs(EngineCoreOutputs):
    outputs: list[OmniEngineCoreOutput] = []
