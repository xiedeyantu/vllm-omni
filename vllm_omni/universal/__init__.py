# SPDX-License-Identifier: Apache-2.0

"""
vLLM-Omni Universal Stage: Non-blocking multi-operator processing pipeline.

Provides:
- UniversalEngine: Non-blocking operator execution
- UniversalStage: Async stage with MessageQueue-based worker pool
- Result collector loop for async result handling
"""

from vllm_omni.universal.engine import UniversalEngine
from vllm_omni.universal.engine_scheduler import UniversalEngineScheduler
from vllm_omni.universal.stage import OmniUniversal
from vllm_omni.universal.worker_process import universal_worker_process

__all__ = [
    "UniversalEngine",
    "UniversalEngineScheduler",
    "OmniUniversal",
    "universal_worker_process",
]
