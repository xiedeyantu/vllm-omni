# SPDX-License-Identifier: Apache-2.0

"""
UniversalStage Worker Process: Engine execution in child process.
"""

import multiprocessing as mp
import time
from typing import Any, Dict
import zmq
from vllm_omni.logger import init_logger
from vllm_omni.universal.data import UniversalOutput
from vllm_omni.universal.zmq_queue import ZmqTaskQueue

# diffusion-specific types used by the worker implementation
from vllm_omni.diffusion.data import OmniDiffusionConfig, DiffusionOutput
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

logger = init_logger(__name__)


def universal_worker_process(
    worker_id: int,
    req_mq_handle: str,
    res_mq_handle: str,
    engine_class: type,
    engine_config: Dict[str, Any],
) -> None:
    """
    Worker process main function.
    
    Responsibilities:
    1. Attach to MessageQueues using handles
    2. Initialize engine
    3. Loop: dequeue requests, execute engine.step(), enqueue results
    4. Handle shutdown gracefully
    
    Args:
        worker_id: Worker process ID
        req_mq_handle: Serialized handle for request MessageQueue
        res_mq_handle: Serialized handle for result MessageQueue
        engine_class: Class to instantiate as engine
        engine_config: Config dict for engine initialization
    """
    logger = init_logger(__name__ + f".worker-{worker_id}")
    
    # Attach to task/result queues (ZMQ-only for this worker)
    try:
        if not (isinstance(req_mq_handle, dict) and "addr" in req_mq_handle):
            raise ValueError("req_mq_handle must be a ZMQ handle dict with 'addr'")
        req_mq = ZmqTaskQueue.create_from_handle(req_mq_handle)

        if not (isinstance(res_mq_handle, dict) and "addr" in res_mq_handle):
            raise ValueError("res_mq_handle must be a ZMQ handle dict with 'addr'")
        # Workers push results to scheduler's result reader addr
        res_mq = ZmqTaskQueue.create_writer_from_handle(res_mq_handle)
        logger.info(f"Worker-{worker_id} attached to ZMQ task/result queues")
    except Exception as e:
        logger.error(f"Failed to attach MessageQueues: {e}", exc_info=True)
        return

    # Initialize engine
    try:
        engine = engine_class(**engine_config)
        logger.info(f"Worker-{worker_id} engine initialized: {engine_class.__name__}")
    except Exception as e:
        logger.error(f"Engine init failed in worker-{worker_id}: {e}", exc_info=True)
        try:
            res_mq.enqueue({
                "tag": "__init_error__",
                "req_id": None,
                "payload": str(e),
            })
        except Exception:
            pass
        return

    try:
        logger.info(f"Worker-{worker_id} entering main loop")
        while True:
            try:
                msg = req_mq.dequeue()
            except Exception as e:
                logger.debug(f"Worker-{worker_id} dequeue error: {e}")
                time.sleep(0.01)
                continue

            if msg is None:
                logger.info(f"Worker-{worker_id} received None, breaking loop")
                break

            # Check for shutdown signal
            if msg.get("shutdown"):
                logger.info(f"Worker-{worker_id} received shutdown signal")
                break

            req_id = msg.get("req_id")
            engine_inputs = msg.get("engine_inputs")
            sampling_params = msg.get("sampling_params")

            logger.debug(f"Worker-{worker_id} processing request {req_id}")

            try:
                # Execute engine step (blocking)
                outputs = engine.step(engine_inputs, sampling_params)
                
                # Send result back
                try:
                    res_mq.enqueue({
                        "tag": "__ok__",
                        "req_id": req_id,
                        "payload": outputs,
                    })
                    logger.debug(f"Worker-{worker_id} enqueued result for {req_id}")
                except Exception as e:
                    logger.error(f"Worker-{worker_id} failed to enqueue result: {e}")

            except Exception as e:
                logger.error(f"Worker-{worker_id} step error for {req_id}: {e}", exc_info=True)
                try:
                    res_mq.enqueue({
                        "tag": "__error__",
                        "req_id": req_id,
                        "payload": str(e),
                    })
                except Exception:
                    pass

    except Exception as e:
        logger.error(f"Worker-{worker_id} fatal error: {e}", exc_info=True)

    finally:
        try:
            engine.close()
            logger.info(f"Worker-{worker_id} engine closed")
        except Exception as e:
            logger.warning(f"Worker-{worker_id} engine close error: {e}")
        logger.info(f"Worker-{worker_id} terminated")



class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        gpu_id: int,
        broadcast_handle,
    ):
        self.od_config = od_config

        # Inter-process Communication
        self.context = zmq.Context(io_threads=2)

        # Initialize reader from handle. This worker expects a ZMQ handle dict
        # with an 'addr' key and will create a PULL socket connected to that addr.
        if isinstance(broadcast_handle, dict) and "addr" in broadcast_handle:
            self.mq = ZmqTaskQueue.create_from_handle(broadcast_handle)
            self._use_zmq = True
        else:
            raise ValueError("broadcast_handle must be a ZMQ handle dict with 'addr'")

        # Worker no longer creates a MessageQueue for results locally.
        # Result channel (if any) should be provided by the scheduler as a handle
        # and will be attached elsewhere. Keep placeholders for compatibility.
        self.result_mq = None
        self.result_mq_handle = None

        assert od_config.master_port is not None
        self.worker = self._create_worker(gpu_id, od_config)
        self.gpu_id = gpu_id
        self._running = True

    def _create_worker(self, gpu_id: int, od_config: OmniDiffusionConfig) -> DiffusionWorker:
        """Create a worker instance. Override in subclasses for different worker types."""
        return DiffusionWorker(
            local_rank=gpu_id,
            rank=gpu_id,
            od_config=od_config,
        )

    def return_result(self, output: UniversalOutput):
        """Reply to client, only on rank 0."""
        if self.result_mq is not None:
            self.result_mq.enqueue(output)

    def recv_message(self):
        """Receive messages from broadcast queue."""
        return self.mq.dequeue(timeout=None)
   

    def execute_rpc(self, rpc_request: dict) -> tuple[object | None, bool]:
        """Execute an RPC request and indicate whether to reply."""
        method = rpc_request["method"]
        args = rpc_request.get("args", ())
        kwargs = rpc_request.get("kwargs", {})
        output_rank = rpc_request.get("output_rank")
        exec_all_ranks = rpc_request.get("exec_all_ranks", False)

        should_execute = exec_all_ranks or output_rank is None or output_rank == self.gpu_id
        should_reply = (output_rank is None or output_rank == self.gpu_id) and self.result_mq is not None

        if not should_execute:
            return None, False

        try:
            if isinstance(method, str):
                func = getattr(self.worker, method)
                result = func(*args, **kwargs)
            else:
                result = method(self.worker, *args, **kwargs)
            return result, should_reply
        except Exception as e:
            logger.error(f"Error executing RPC: {e}", exc_info=True)
            raise e

    def worker_busy_loop(self) -> None:
        """Main busy loop for Multiprocessing Workers."""
        logger.info(f"Worker {self.gpu_id} ready to receive requests via shared memory")

        while self._running:
            msg = None
            try:
                msg = self.recv_message()
            except Exception as e:
                logger.error(
                    f"Error receiving message in worker loop: {e}",
                    exc_info=True,
                )
                continue

            if msg is None or len(msg) == 0:
                logger.warning("Worker %s: Received empty payload, ignoring", self.gpu_id)
                continue

            # Route message based on type
            if isinstance(msg, dict) and msg.get("type") == "rpc":
                try:
                    result, should_reply = self.execute_rpc(msg)
                    if should_reply:
                        self.return_result(result)
                except Exception as e:
                    logger.error(f"Error processing RPC: {e}", exc_info=True)
                    if self.result_mq is not None:
                        self.return_result(UniversalOutput(error=str(e)))

            elif isinstance(msg, dict) and msg.get("type") == "shutdown":
                logger.info("Worker %s: Received shutdown message", self.gpu_id)
                self._running = False
                continue

            else:
                # Handle generation request
                try:
                    output = self.worker.execute_model(msg, self.od_config)
                except Exception as e:
                    logger.error(
                        f"Error executing forward in event loop: {e}",
                        exc_info=True,
                    )
                    output = UniversalOutput(error=str(e))

                try:
                    self.return_result(output)
                except zmq.ZMQError as e:
                    logger.error(f"ZMQ error sending reply: {e}")
                    continue

        logger.info("event loop terminated.")
        try:
            self.worker.shutdown()
        except Exception as exc:
            logger.warning("Worker %s: Shutdown encountered an error: %s", self.gpu_id, exc)
        self.context.term()

    @staticmethod
    def worker_main(
        rank: int,
        od_config: OmniDiffusionConfig,
        pipe_writer: mp.connection.Connection,
        broadcast_handle,
    ) -> None:
        """Worker initialization and execution loops."""
        from vllm_omni.plugins import load_omni_general_plugins

        load_omni_general_plugins()
        worker_proc = WorkerProc(
            od_config,
            gpu_id=rank,
            broadcast_handle=broadcast_handle,
        )
        logger.info(f"Worker {rank}: Scheduler loop started.")
        pipe_writer.send(
            {
                "status": "ready",
                "result_handle": worker_proc.result_mq_handle if rank == 0 else None,
            }
        )
        worker_proc.worker_busy_loop()
        logger.info(f"Worker {rank}: Shutdown complete.")
