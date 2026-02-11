# SPDX-License-Identifier: Apache-2.0

"""
UniversalStage: Multi-worker universal processing stage with MessageQueue distribution.

Architecture:
- UniversalScheduler manages request queueing
- Multiple worker processes execute via UniversalEngine
- MessageQueue distributes requests and collects results
- Async result collection via background thread and asyncio futures
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid
from typing import Any, Iterable

from vllm import SamplingParams

from vllm_omni.logger import init_logger
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.inputs.data import OmniPromptType
from vllm_omni.universal.data import OmniUniversalConfig
from vllm_omni.universal.engine import UniversalEngine
from vllm_omni.universal.request import OmniUniversalRequest

logger = init_logger(__name__)


class AsyncOmniUniversal:
    def __init__(
        self,
        model: str,
        ou_config: OmniUniversalConfig | None = None,
        **kwargs: Any,
    ):
        self.model = model

        stage_id = kwargs.get("stage_id")
        engine_input_source = kwargs.get("engine_input_source")

        # Build config
        if ou_config is None:
            ou_config = OmniUniversalConfig.from_kwargs(model=model, **kwargs)
        elif isinstance(ou_config, dict):
            # If config is dict, check it too (priority to kwargs if both exist)
            if stage_id is None:
                stage_id = ou_config.get("stage_id")
            if engine_input_source is None:
                engine_input_source = ou_config.get("engine_input_source")
            ou_config = OmniUniversalConfig.from_kwargs(**ou_config)

        self.ou_config = ou_config

        # Inject stage info into omni_kv_config if present
        if stage_id is not None:
            self.ou_config.omni_kv_config.setdefault("stage_id", stage_id)
        if engine_input_source is not None:
            self.ou_config.omni_kv_config.setdefault("engine_input_source", engine_input_source)

        # Initialize engine
        self.engine: UniversalEngine = UniversalEngine.make_engine(ou_config)

        # Thread pool for running sync engine in async context
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._closed = False

        logger.info("AsyncOmniDiffusion initialized with model: %s", model)

    async def generate(
        self,
        prompt: OmniPromptType,
        sampling_params: Any | None,
        request_id: str | None = None,
        lora_request: Any | None = None,
    ) -> OmniRequestOutput:
        if request_id is None:
            request_id = f"univ-{uuid.uuid4().hex[:16]}"

        request = OmniUniversalRequest(
            prompts=[prompt],
            request_ids=[request_id], # 
        )

        logger.debug("Starting generation for request %s", request_id)

        # Run engine in thread pool
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(self._executor, self.engine.step, request)

            # schedule the blocking wait in the threadpool
            scheduler = self.engine.get_scheduler()
            if scheduler is None:
                raise RuntimeError("Could not locate scheduler to read results from")

            result = await loop.run_in_executor(self._executor, self._wait_for_result, scheduler, request.request_ids[0])
        except Exception as e:
            logger.error("Generation failed for request %s: %s", request_id, e)
            raise RuntimeError(f"Universal generation failed: {e}") from e

        # Normalize result to OmniRequestOutput
        if isinstance(result, OmniRequestOutput):
            if not getattr(result, "request_id", None):
                result.request_id = request_id
            return result
        else:
            # Wrap raw payload into OmniRequestOutput
            return OmniRequestOutput(
                request_id=request_id,
                finished=True,
                final_output_type="json",
                multimodal_output={"payload": result},
            )

    def _wait_for_result(self, sched, req_id: str):
        """Blocking helper to read scheduler result queue for a specific request id.

        This runs inside the threadpool so it can block on reader.dequeue().
        It supports both the new ZMQ dict-style messages (tag/req_id/payload)
        and the legacy OmniRequestOutput objects.
        """
        # Prefer ZMQ-based reader if available, fall back to legacy result_mq
        reader = getattr(sched, "result_reader", None) or getattr(sched, "result_mq", None)
        if reader is None:
            raise RuntimeError("Result queue not initialized on scheduler")

        while True:
            # reader.dequeue is blocking; this runs in the executor
            msg = reader.dequeue()

            # New-style workers send dicts with tags
            if isinstance(msg, dict):
                # If payload carries the request id, match it
                if msg.get("req_id") == req_id:
                    tag = msg.get("tag")
                    payload = msg.get("payload")
                    if tag in ("__error__", "__init_error__"):
                        raise RuntimeError(f"Worker error: {payload}")
                    return payload

            # Legacy path: direct OmniRequestOutput returned
            if hasattr(msg, "request_id") and getattr(msg, "request_id") == req_id:
                return msg

            # Otherwise ignore unrelated messages and continue waiting

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        try:
            self.engine.close()
        except Exception as e:
            logger.warning("Error closing universal engine: %s", e)

        try:
            self._executor.shutdown(wait=False)
        except Exception as e:
            logger.warning("Error shutting down executor: %s", e)

        logger.info("AsyncOmniUniversal closed")

    def shutdown(self) -> None:
        """Alias for close() method."""
        self.close()

    def __del__(self) -> None:
        """Best-effort cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass

    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort a request."""
        self.engine.abort(request_id)

    @property
    def is_running(self) -> bool:
        """Check if the engine is running."""
        return not self._closed

    @property
    def is_stopped(self) -> bool:
        """Check if the engine is stopped."""
        return self._closed

