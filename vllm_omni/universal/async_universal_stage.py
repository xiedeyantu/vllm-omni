import asyncio
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence, List, AsyncGenerator
from vllm_omni.universal.engine import UniversalEngine, UniversalEngineRequest
from vllm_omni.universal.universal_stage import UniversalStageConfig
from vllm_omni.outputs import OmniRequestOutput, UniversalStageOutput
from vllm_omni.inputs.data import OmniPromptType

class AsyncUniversalStage:
    """
    Async entry point for vLLM-Omni universal stage inference.
    """
    def __init__(self, config: UniversalStageConfig):
        self.config = config
        self.stage_id = config.stage_id
        self.engine_input_source = config.engine_input_source
        # Initialize engine
        self.engine: UniversalEngine = UniversalEngine.make_engine(config)
        # Thread pool for running sync engine in async context
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._closed = False

    async def generate(
        self,
        prompts: Sequence[OmniPromptType],
        sampling_params: Any = None,
        request_ids: List[str] = None,
        **kwargs
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Generate outputs asynchronously.
        
        Matches the async pattern of AsyncOmniDiffusion.
        """
        # Normalize prompts to a list
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts]
        else:
            prompts = list(prompts)

        n = len(prompts)
        # Ensure request_ids length matches prompts (use similar format as sync stage)
        if request_ids is None:
            request_ids = [f"{i}_{uuid.uuid4()}" for i in range(n)]
        elif len(request_ids) < n:
            request_ids = list(request_ids) + [f"{len(request_ids) + i}_{uuid.uuid4()}" for i in range(n - len(request_ids))]

        logger = logging.getLogger(__name__)
        
        # Create internal request object for async execution
        internal_request = UniversalEngineRequest(
            prompts=prompts,
            sampling_params=sampling_params,
            request_ids=request_ids
        )
        
        loop = asyncio.get_event_loop()

        # Run the entire batch in the executor to avoid blocking
        try:
            engine_outputs = await loop.run_in_executor(
                self._executor,
                self.engine.step,
                UniversalEngineRequest(
                    prompts=internal_request.prompts,
                    sampling_params=internal_request.sampling_params,
                    request_ids=internal_request.request_ids
                )
            )
        except Exception as e:
            logger.error(f"Universal generation failed: {e}")
            raise RuntimeError(f"Universal generation failed: {e}") from e

        # Wrap engine outputs into OmniRequestOutput (yield as stream)
        for rid, engine_output in zip(internal_request.request_ids, engine_outputs):
            try:
                # Convert engine output to UniversalStageOutput, then to OmniRequestOutput
                us_output = UniversalStageOutput.from_engine_output(
                    request_id=rid,
                    stage_id=self.stage_id,
                    engine_output=engine_output,
                    execution_time_ms=0,  # Time already accumulated in sync execution
                )
                ro = us_output.to_omni_request_output()
            except Exception as e:
                logger.error(f"Error wrapping output for request {rid}: {e}")
                # Create basic error response
                ro = OmniRequestOutput(request_id=rid, stage_id=self.stage_id, finished=True)

            yield ro
    # Lifecycle helpers expected by omni async worker
    async def reset_mm_cache(self) -> None:
        """No-op for UniversalStage (kept for API compatibility)."""
        return None

    async def is_tracing_enabled(self) -> bool:
        """UniversalStage does not perform tracing via vLLM tracer; return False."""
        return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

    def shutdown(self) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

