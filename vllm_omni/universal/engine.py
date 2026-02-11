# SPDX-License-Identifier: Apache-2.0

"""
UniversalEngine: Non-blocking engine for processing requests via operator chains.

Key design:
- step() returns None (non-blocking)
- Internally manages scheduler to track requests and results
- Operators execute one at a time within step()
"""

import os
import time
from typing import Any, Iterable, Optional

from vllm.logger import init_logger

from vllm_omni.diffusion.registry import register_dummy_run_for_omni_universal
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.universal.data import OmniUniversalConfig
from vllm_omni.universal.excutor import UniversalExecutor
from vllm_omni.universal.request import OmniUniversalRequest

logger = init_logger(__name__)


class UniversalEngine:

    def __init__(self, ou_config: OmniUniversalConfig):
        """Initialize the universal engine.

        Args:
            config: The configuration for the universal engine.
        """
        self.ou_config = ou_config

        executor_class = UniversalExecutor.get_class(ou_config)
        self.executor = executor_class(ou_config)

        try:
            _dummy_run = register_dummy_run_for_omni_universal(ou_config)
            _dummy_run()
        except Exception as e:
            logger.error(f"Dummy run failed: {e}")
            self.close()
            raise e

    def step(self, request: OmniUniversalRequest) -> None:
        self.add_req(request)

    @staticmethod
    def make_engine(config: OmniUniversalConfig) -> "UniversalEngine":
        """Factory method to create a UniversalEngine instance.

        Args:
            config: The configuration for the universal engine.

        Returns:
            An instance of UniversalEngine.
        """
        return UniversalEngine(config)

    def add_req(self, request: OmniUniversalRequest):
        self.executor.add_req(request)


    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Call a method on worker processes and get results immediately.

        Args:
            method: The method name (str) to execute on workers
            timeout: Optional timeout in seconds
            args: Positional arguments for the method
            kwargs: Keyword arguments for the method
            unique_reply_rank: If set, only get reply from this rank

        Returns:
            Single result if unique_reply_rank is provided, otherwise list of results
        """
        assert isinstance(method, str), "Only string method names are supported for now"
        return self.executor.collective_rpc(
            method=method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
            unique_reply_rank=unique_reply_rank,
        )

    def get_scheduler(self):
        """Return the underlying scheduler instance if available.

        Some executor implementations attach the scheduler as
        `self.executor.scheduler`, while others may expose it as
        `self.scheduler` directly on the engine. This helper centralizes
        that lookup and avoids callers having to use multiple getattr
        calls.
        """
        # Strict implementation: expect scheduler to be attached to executor.
        if not hasattr(self, "executor"):
            raise RuntimeError("Executor not initialized on engine")

        if not hasattr(self.executor, "scheduler"):
            raise RuntimeError("Executor does not expose a 'scheduler' attribute")

        return self.executor.scheduler
    
    def _process_request(self, req) -> OmniRequestOutput:
        """
        Process a single request through the operator chain.
        
        Args:
            req: ScheduledRequest object
            
        Returns:
            OmniRequestOutput with results
        """
        # TODO: Implement actual operator chain execution
        # For now, return a simple output
        
        logger.debug(f"Processing request {req.request_id}")
        
        # Simulate processing
        output = OmniRequestOutput(
            request_id=req.request_id,
            finished=True,
            final_output_type="json",
            multimodal_output={
                "input": req.engine_inputs,
                "sampling_params": req.sampling_params,
                "timestamp": time.time(),
            },
        )
        return output

    def has_result(self, request_id: str) -> bool:
        """Check if a result is ready for this request."""
        # Simple check: scan completed results
        # In production, might maintain a dict for O(1) lookup
        return self.scheduler.has_pending_results()

    def get_result(self, request_id: str) -> Optional[OmniRequestOutput]:
        """Retrieve a completed result (FIFO)."""
        if self.scheduler.has_pending_results():
            result = self.scheduler.get_next_result()
            if result and result.request_id == request_id:
                return result.output
        return None






    def start_profile(self, trace_filename: str | None = None) -> None:
        if trace_filename is None:
            trace_filename = f"universal_trace_{int(time.time())}"

        trace_dir = os.environ.get("VLLM_TORCH_PROFILER_DIR", "./profiles")

        # Expand ~ and ~user, then make absolute (robust against cwd changes)
        trace_dir = os.path.expanduser(trace_dir)
        trace_dir = os.path.abspath(trace_dir)

        try:
            os.makedirs(trace_dir, exist_ok=True)
        except OSError as exc:
            logger.error(f"Failed to create profiler directory {trace_dir}: {exc}")
            raise

        # Build final template path (without rank or extension — torch.profiler appends those)
        full_template = os.path.join(trace_dir, trace_filename)

        expected_pattern = f"{full_template}*.json"
        logger.info(f"Starting universal profiling → {expected_pattern}")

        # Also log the absolute directory once (useful in multi-node or containers)
        logger.debug(f"Profiler output directory: {trace_dir}")

        # Propagate to all workers
        try:
            self.collective_rpc(method="start_profile", args=(full_template,))
        except Exception as e:
            logger.error("Failed to start profiling on workers", exc_info=True)
            raise RuntimeError(f"Could not start profiler: {e}") from e

    def stop_profile(self) -> dict:
        """
        Stop profiling on all workers and collect the final trace/table paths.

        The worker (torch_profiler.py) now handles trace export, compression to .gz,
        and deletion of the original .json file. This method only collects and
        reports the paths returned by the workers.

        Returns:
            dict with keys:
            - "traces": list of final trace file paths (usually .json.gz)
            - "tables": list of table strings (one per rank)
        """
        logger.info("Stopping diffusion profiling and collecting results...")

        try:
            # Give worker enough time — export + compression + table can be slow
            results = self.collective_rpc(method="stop_profile", timeout=60000)
        except Exception:
            logger.error("Failed to stop profiling on workers", exc_info=True)
            return {"traces": [], "tables": []}

        output_files = {"traces": [], "tables": []}
        successful_traces = 0

        if not results:
            logger.warning("No profiling results returned from any rank")
            return output_files

        for rank, res in enumerate(results):
            if not isinstance(res, dict):
                logger.warning(f"Rank {rank}: invalid result format (got {type(res)})")
                continue

            # 1. Trace file — should be .json.gz if compression succeeded
            trace_path = res.get("trace")
            if trace_path:
                # We trust the worker — it created/compressed the file
                logger.info(f"[Rank {rank}] Final trace: {trace_path}")
                output_files["traces"].append(trace_path)
                successful_traces += 1

                # Optional: warn if path looks suspicious (e.g. still .json)
                if not trace_path.endswith((".json.gz", ".json")):
                    logger.warning(f"Rank {rank}: unusual trace path extension: {trace_path}")

            # 2. Table file — plain text
            table = res.get("table")
            if table:
                output_files["tables"].append(table)

        # Final summary logging
        num_ranks = len(results)
        if successful_traces > 0:
            final_paths_str = ", ".join(output_files["traces"][:3])
            if len(output_files["traces"]) > 3:
                final_paths_str += f" ... (+{len(output_files['traces']) - 3} more)"

            logger.info(
                f"Profiling stopped. Collected {successful_traces} trace file(s) "
                f"from {num_ranks} rank(s). "
                f"Final trace paths: {final_paths_str}"
            )
        elif output_files["traces"]:
            logger.info(
                f"Profiling stopped but no traces were successfully collected. "
                f"Reported paths: {', '.join(output_files['traces'][:3])}"
                f"{' ...' if len(output_files['traces']) > 3 else ''}"
            )
        else:
            logger.info("Profiling stopped — no trace files were collected from any rank.")

        if output_files["tables"]:
            logger.debug(f"Collected {len(output_files['tables'])} profiling table(s)")

        return output_files

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Call a method on worker processes and get results immediately.

        Args:
            method: The method name (str) to execute on workers
            timeout: Optional timeout in seconds
            args: Positional arguments for the method
            kwargs: Keyword arguments for the method
            unique_reply_rank: If set, only get reply from this rank

        Returns:
            Single result if unique_reply_rank is provided, otherwise list of results
        """
        assert isinstance(method, str), "Only string method names are supported for now"
        return self.executor.collective_rpc(
            method=method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
            unique_reply_rank=unique_reply_rank,
        )

    def close(self) -> None:
        if hasattr(self, "executor"):
            self.executor.shutdown()

    def abort(self, request_id: str | Iterable[str]) -> None:
        # TODO implement it
        logger.warning("DiffusionEngine abort is not implemented yet")
        pass
