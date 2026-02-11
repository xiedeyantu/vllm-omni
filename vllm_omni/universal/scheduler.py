# SPDX-License-Identifier: Apache-2.0

"""
UniversalScheduler: Request scheduling and queueing for UniversalStage.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List
from vllm_omni.logger import init_logger
from vllm_omni.universal.data import OmniUniversalConfig, UniversalOutput
import zmq
from vllm_omni.universal.zmq_queue import ZmqTaskQueue
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

from vllm_omni.universal.request import OmniUniversalRequest


logger = init_logger(__name__)


@dataclass
class UniversalRequest:
    """Single request to be processed by UniversalEngine."""
    request_id: str
    engine_inputs: Any  # Could be prompt, dict, list, etc.
    sampling_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "engine_inputs": self.engine_inputs,
            "sampling_params": self.sampling_params,
            "metadata": self.metadata,
        }


class UniversalScheduler:
    def initialize(self, ou_config: OmniUniversalConfig):
        existing_mq = getattr(self, "mq", None)
        if existing_mq is not None and not existing_mq.closed:
            logger.warning("SyncSchedulerClient is already initialized. Re-initializing.")
            self.close()

        self.num_workers = ou_config.num_gpus
        self.ou_config = ou_config

        # Create a ZMQ task queue (scheduler binds a PUSH socket; workers will PULL)
        bind_ip = ou_config.host or "0.0.0.0"
        self.task_writer = ZmqTaskQueue.create_writer(bind_ip=bind_ip)
        self.task_handle = self.task_writer.export_handle()

        # Create a ZMQ result reader (scheduler binds a PULL socket; workers PUSH results)
        self.result_reader = ZmqTaskQueue.create_reader_bind(bind_ip=bind_ip)
        self.result_handle = self.result_reader.export_handle()

        # Keep optional legacy MessageQueue for other control messages if needed
        self.mq = MessageQueue(
            n_reader=self.num_workers,
            n_local_reader=self.num_workers,
            local_reader_ranks=list(range(self.num_workers)),
        )

    def initialize_result_queue(self, handle):
        # Legacy support: accept a MessageQueue handle and set up result_mq
        self.result_mq = MessageQueue.create_from_handle(handle, rank=0)
        logger.info("SyncScheduler initialized legacy result MessageQueue")

    def get_broadcast_handle(self):
        # Return the control/messaging handle (legacy)
        return self.mq.export_handle()
    
    def get_mq_handle(self):
        return self.mq.export_handle()

    def get_task_handle(self) -> Dict[str, str]:
        """Return ZMQ handle for task queue (workers should create PULL sockets)."""
        return self.task_handle

    def get_result_handle(self) -> Dict[str, str]:
        """Return ZMQ handle for result queue (workers should create PUSH sockets)."""
        return self.result_handle


    def add_req(self, request: OmniUniversalRequest) -> UniversalOutput:
        """Sends a request to the scheduler and waits for the response."""
        try:
            # Prepare RPC request for generation
            rpc_request = {
                "type": "rpc",
                "method": "generate",
                "args": (request,),
                "kwargs": {},
                "output_rank": 0,
                "exec_all_ranks": True,
            }

            # Broadcast RPC request to all workers
            self.mq.enqueue(rpc_request)
            # Wait for result from Rank 0 (or whoever sends it)

            if self.result_mq is None:
                raise RuntimeError("Result queue not initialized")

            output = self.result_mq.dequeue()
            # {"status": "error", "error": str(e)}
            if isinstance(output, dict) and output.get("status") == "error":
                raise RuntimeError("worker error")
            return output
        except zmq.error.Again:
            logger.error("Timeout waiting for response from scheduler.")
            raise TimeoutError("Scheduler did not respond in time.")

    def close(self):
        """Closes the socket and terminates the context."""
        if hasattr(self, "context"):
            self.context.term()
        self.context = None
        self.mq = None
        self.result_mq = None



    def __init__(self):
        self._pending_reqs: List[UniversalRequest] = []
        self._request_count = 0

    def add_request(
        self,
        request_id: str,
        engine_inputs: Any,
        sampling_params: Dict[str, Any] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Add a request to the queue."""
        req = UniversalRequest(
            request_id=request_id,
            engine_inputs=engine_inputs,
            sampling_params=sampling_params or {},
            metadata=metadata or {},
        )
        self._pending_reqs.append(req)
        self._request_count += 1
        logger.debug(f"Added request {request_id}, total pending: {len(self._pending_reqs)}")

    def has_pending_requests(self) -> bool:
        """Check if there are pending requests."""
        return len(self._pending_reqs) > 0

    def get_next_request(self) -> UniversalRequest | None:
        """Get the next request from the queue (FIFO)."""
        if self._pending_reqs:
            req = self._pending_reqs.pop(0)
            logger.debug(f"Dequeued request {req.request_id}")
            return req
        return None

    def get_batch(self, batch_size: int = 1) -> List[UniversalRequest]:
        """Get a batch of requests."""
        batch = []
        for _ in range(min(batch_size, len(self._pending_reqs))):
            req = self._pending_reqs.pop(0)
            batch.append(req)
        return batch

    def num_pending(self) -> int:
        """Get number of pending requests."""
        return len(self._pending_reqs)

    def stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "pending_requests": len(self._pending_reqs),
            "total_requests": self._request_count,
        }
