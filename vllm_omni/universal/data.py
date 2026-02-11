# adapted from sglang and fastvideo
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import os
import random
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Any

import torch
from pydantic import model_validator
from typing_extensions import Self
from vllm.config.utils import config
from vllm.logger import init_logger

from vllm_omni.universal.network_utils import is_port_available

logger = init_logger(__name__)


@dataclass
class OmniUniversalConfig:
    # Model and path configuration (for convenience)
    model: str | None = None

    model_class_name: str | None = None

    dtype: torch.dtype = torch.bfloat16

    # Running mode
    # mode: ExecutionMode = ExecutionMode.INFERENCE

    # Workload type
    # workload_type: WorkloadType = WorkloadType.T2V

    # Distributed executor backend
    distributed_executor_backend: str = "mp"
    nccl_port: int | None = None

    # HuggingFace specific parameters
    trust_remote_code: bool = False
    revision: str | None = None

    num_gpus: int | None = None

    # pipeline_config: PipelineConfig = field(default_factory=PipelineConfig, repr=False)

    output_type: str = "pil"

    # Enable sleep mode
    enable_sleep_mode: bool = False

    # V-MoBA parameters
    moba_config_path: str | None = None
    # moba_config: dict[str, Any] = field(default_factory=dict)

    # Master port for distributed inference
    # TODO: do not hard code
    master_port: int | None = None

    scheduler_port: int = 5555

    # Stage verification
    enable_stage_verification: bool = True

    # Prompt text file for batch processing
    prompt_file_path: str | None = None

    # model paths for correct deallocation
    model_paths: dict[str, str] = field(default_factory=dict)

    # # DMD parameters
    # dmd_denoising_steps: List[int] | None = field(default=None)

    # Logging
    log_level: str = "info"

    # Omni configuration (injected from stage config)
    omni_kv_config: dict[str, Any] = field(default_factory=dict)

    def settle_port(self, port: int, port_inc: int = 42, max_attempts: int = 100) -> int:
        """
        Find an available port with retry logic.

        Args:
            port: Initial port to check
            port_inc: Port increment for each attempt
            max_attempts: Maximum number of attempts to find an available port

        Returns:
            An available port number

        Raises:
            RuntimeError: If no available port is found after max_attempts
        """
        attempts = 0
        original_port = port

        while attempts < max_attempts:
            if is_port_available(port):
                if attempts > 0:
                    logger.info(f"Port {original_port} was unavailable, using port {port} instead")
                return port

            attempts += 1
            if port < 60000:
                port += port_inc
            else:
                # Wrap around with randomization to avoid collision
                port = 5000 + random.randint(0, 1000)

        raise RuntimeError(
            f"Failed to find available port after {max_attempts} attempts (started from port {original_port})"
        )

    def __post_init__(self):
        # TODO: remove hard code
        initial_master_port = (self.master_port or 30005) + random.randint(0, 100)
        self.master_port = self.settle_port(initial_master_port, 37)

        if self.num_gpus is None:
            if self.parallel_config is not None:
                self.num_gpus = self.parallel_config.world_size
            else:
                self.num_gpus = 1

        if self.num_gpus < self.parallel_config.world_size:
            raise ValueError(
                f"num_gpus ({self.num_gpus}) < parallel_config.world_size ({self.parallel_config.world_size})"
            )

        # Convert string dtype to torch.dtype if needed
        if isinstance(self.dtype, str):
            dtype_map = {
                "auto": torch.bfloat16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
                "float": torch.float32,
            }
            dtype_lower = self.dtype.lower()
            if dtype_lower in dtype_map:
                self.dtype = dtype_map[dtype_lower]
            else:
                logger.warning(f"Unknown dtype string '{self.dtype}', defaulting to bfloat16")
                self.dtype = torch.bfloat16

    def update_multimodal_support(self) -> None:
        self.supports_multimodal_inputs = self.model_class_name in {"QwenImageEditPlusPipeline"}

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "OmniUniversalConfig":
        # Backwards-compatibility: older callers may use a Universal-specific
        # "static_lora_scale" kwarg. Normalize it to the canonical "lora_scale"
        # before constructing the dataclass to avoid TypeError on unknown fields.
        if "static_lora_scale" in kwargs:
            if "lora_scale" not in kwargs:
                kwargs["lora_scale"] = kwargs["static_lora_scale"]
            kwargs.pop("static_lora_scale", None)

        # Check environment variable as fallback for cache_backend
        # Support both old Universal_CACHE_ADAPTER and new Universal_CACHE_BACKEND for backwards compatibility
        if "cache_backend" not in kwargs:
            cache_backend = os.environ.get("Universal_CACHE_BACKEND") or os.environ.get("Universal_CACHE_ADAPTER")
            kwargs["cache_backend"] = cache_backend.lower() if cache_backend else "none"

        # Filter kwargs to only include valid fields
        valid_fields = {f.name for f in fields(cls)}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

        return cls(**filtered_kwargs)


@dataclass
class UniversalOutput:
    """
    Final output (after pipeline completion)
    """

    output: torch.Tensor | None = None
    error: str | None = None

    post_process_func: Callable[..., Any] | None = None

    # logged timings info, directly from Req.timings
    # timings: Optional["RequestTimings"] = None


# Special message broadcast via scheduler queues to signal worker shutdown.
SHUTDOWN_MESSAGE = {"type": "shutdown"}
