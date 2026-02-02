import zmq
import pickle
import time
from typing import Any, Dict, Optional, Tuple
from .base import OmniConnectorBase
from ..utils.logging import get_connector_logger

logger = get_connector_logger(__name__)

class ZmqConnector(OmniConnectorBase):
    """
    Connector that uses ZMQ for stage-to-stage communication.
    Supports PUSH/PULL pattern for asynchronous processing.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stage_id = config.get("stage_id", -1)
        self.role = config.get("role", "sender") # "sender" (PUSH) or "receiver" (PULL)
        self.address = config.get("address", "tcp://127.0.0.1:5555")
        self.bind = config.get("bind", False)
        
        self.context = zmq.Context()
        if self.role == "sender":
            self.socket = self.context.socket(zmq.PUSH)
        else:
            self.socket = self.context.socket(zmq.PULL)
            
        if self.bind:
            self.socket.bind(self.address)
            logger.info(f"[ZmqConnector] Bound to {self.address} as {self.role}")
        else:
            self.socket.connect(self.address)
            logger.info(f"[ZmqConnector] Connected to {self.address} as {self.role}")

    def put(self, from_stage: str, to_stage: str, put_key: str, data: Any) -> Tuple[bool, int, Optional[Dict[str, Any]]]:
        try:
            payload = pickle.dumps((put_key, data))
            size = len(payload)
            # Send non-blocking or with a short timeout? 
            # Usual PUSH will block if no receivers are connected and HWM is reached.
            self.socket.send(payload)
            return True, size, {"zmq_key": put_key}
        except Exception as e:
            logger.error(f"[ZmqConnector] Put failed: {e}")
            return False, 0, None

    def get(self, from_stage: str, to_stage: str, get_key: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[Tuple[Any, int]]:
        try:
            # Note: ZMQ PULL doesn't let us "get by key". It's a stream.
            # In a pipeline, the next message received IS the next task.
            # If the orchestrator is still managing the in_q, we might have a mismatch
            # if we use ZMQ for data and mp.Queue for signaling.
            # However, if we use ZMQ for EVERYTHING between these stages, then it works.
            
            # For simplicity in this "Universal" stage, we assume sequential processing or
            # that the key is checked.
            
            payload = self.socket.recv()
            recv_key, data = pickle.loads(payload)
            size = len(payload)
            
            if get_key and recv_key != get_key:
                logger.warning(f"[ZmqConnector] Received key {recv_key} but expected {get_key}. This might happen in multi-stage pipelines.")
            
            return data, size
        except Exception as e:
            logger.error(f"[ZmqConnector] Get failed: {e}")
            return None

    def cleanup(self, request_id: str) -> None:
        pass

    def shutdown(self) -> None:
        self.socket.close()
        self.context.term()
