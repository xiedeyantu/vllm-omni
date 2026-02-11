"""
Lightweight ZMQ PUSH/PULL task queue wrapper for UniversalStage.

This module provides a minimal, MessageQueue-like wrapper around ZMQ
PUSH/PULL sockets for competing-consumer semantics (each task is
consumed by exactly one worker). It also exposes a small serializable
`ZmqHandle` used to pass endpoint addresses between processes.

Design notes:
- Scheduler binds:
  - task writer: PUSH bind -> workers PULL connect (tasks)
  - result reader: PULL bind -> workers PUSH connect (results)
- Worker uses handles to connect its PULL (tasks) and PUSH (results).

This is intentionally small; it doesn't implement persistence/acks.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Optional

import zmq

from vllm_omni.logger import init_logger

logger = init_logger(__name__)


@dataclass
class ZmqHandle:
    """Serializable handle describing a ZMQ endpoint."""

    addr: str
    # allow future extensions (ipv6 flag, metadata)
    remote_addr_ipv6: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"addr": self.addr, "remote_addr_ipv6": self.remote_addr_ipv6}

    @classmethod
    def from_any(cls, obj: Any) -> "ZmqHandle":
        if isinstance(obj, ZmqHandle):
            return obj
        if isinstance(obj, dict):
            addr = obj.get("addr")
            if not addr:
                raise ValueError("Invalid handle dict: missing 'addr'")
            return cls(addr=addr, remote_addr_ipv6=bool(obj.get("remote_addr_ipv6", False)))
        raise TypeError("Unsupported handle type: %r" % (type(obj),))


class ZmqTaskQueue:
    """Simple ZMQ PUSH/PULL wrapper providing a handle-based API.

    The API mirrors the minimal parts of vLLM's MessageQueue used by
    the scheduler/worker code: export_handle(), create_from_handle(),
    enqueue(), dequeue(), and close().
    """

    def __init__(self) -> None:
        # internal constructor, prefer factory methods below
        self._ctx = zmq.Context.instance()
        self._socket: Optional[zmq.Socket] = None
        self._is_writer = False
        self._addr: Optional[str] = None

    # --- Writer factory -------------------------------------------------
    @staticmethod
    def create_writer(bind_ip: str = "0.0.0.0", port: Optional[int] = None) -> "ZmqTaskQueue":
        """Create a writer (PUSH socket) and bind to the given ip/port.

        If port is None a random free port will be chosen.
        Returns a ZmqTaskQueue instance and you can call `export_handle()`
        to obtain a small handle dict to send to readers.
        """
        q = ZmqTaskQueue()
        q._is_writer = True
        q._socket = q._ctx.socket(zmq.PUSH)
        if port is None:
            port = q._socket.bind_to_random_port(f"tcp://{bind_ip}")
        else:
            addr = f"tcp://{bind_ip}:{port}"
            q._socket.bind(addr)
        q._addr = f"tcp://{bind_ip}:{port}"
        logger.debug("ZmqTaskQueue writer bound to %s", q._addr)
        return q

    # --- Reader construction --------------------------------------------
    @staticmethod
    def create_from_handle(handle: Dict[str, Any]) -> "ZmqTaskQueue":
        """Create a reader (PULL socket) from a handle produced by a writer.

        The handle is expected to be the dict returned by `export_handle()`.
        """
        # accept either a ZmqHandle or a dict
        h = ZmqHandle.from_any(handle)
        q = ZmqTaskQueue()
        q._is_writer = False
        q._socket = q._ctx.socket(zmq.PULL)
        q._addr = h.addr
        q._socket.connect(h.addr)
        logger.debug("ZmqTaskQueue reader connected to %s", h.addr)
        return q

    @staticmethod
    def create_writer_from_handle(handle: Dict[str, Any]) -> "ZmqTaskQueue":
        """Create a writer (PUSH socket) that CONNECTs to an existing address.

        This is useful when the *reader* has bound a PULL socket and exported
        its handle (addr). Callers creating writers that should connect to
        that addr can use this helper.
        """
        h = ZmqHandle.from_any(handle)
        q = ZmqTaskQueue()
        q._is_writer = True
        q._socket = q._ctx.socket(zmq.PUSH)
        q._addr = h.addr
        q._socket.connect(h.addr)
        logger.debug("ZmqTaskQueue writer connected to %s", h.addr)
        return q

    @staticmethod
    def create_reader_bind(bind_ip: str = "0.0.0.0", port: Optional[int] = None) -> "ZmqTaskQueue":
        """Create a reader (PULL) that BINDs to an address and export its handle.

        This is useful for result queues where the scheduler wants to receive
        PUSHes from workers. Workers will create a PUSH socket that connects
        to the returned handle via `create_writer_from_handle`.
        """
        q = ZmqTaskQueue()
        q._is_writer = False
        q._socket = q._ctx.socket(zmq.PULL)
        if port is None:
            port = q._socket.bind_to_random_port(f"tcp://{bind_ip}")
        else:
            addr = f"tcp://{bind_ip}:{port}"
            q._socket.bind(addr)
        q._addr = f"tcp://{bind_ip}:{port}"
        logger.debug("ZmqTaskQueue reader bound to %s", q._addr)
        return q

    # --- Handle API -----------------------------------------------------
    def export_handle(self) -> ZmqHandle:
        """Return a small serializable handle that readers can use to connect.

        The handle is just a dict with the 'addr' key (tcp://ip:port).
        """
        if self._addr is None:
            raise RuntimeError("No addr available to export")
        return ZmqHandle(addr=self._addr)

    # --- Send/receive --------------------------------------------------
    def enqueue(self, obj: Any) -> None:
        """Send an object to the queue (writer only)."""
        if not self._is_writer or self._socket is None:
            raise RuntimeError("enqueue can only be called on a writer")
        # use pyobj serialization; zmq will use pickle under the hood
        self._socket.send_pyobj(obj, flags=0)

    def dequeue(self, timeout: Optional[float] = None) -> Any:
        """Receive an object from the queue (reader only).

        Args:
            timeout: optional timeout in *milliseconds*. If None, block
                indefinitely. If 0, poll once and raise TimeoutError on no data.
        """
        if self._is_writer or self._socket is None:
            raise RuntimeError("dequeue can only be called on a reader")
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        timeout_ms = None if timeout is None else int(timeout)
        socks = dict(poller.poll(timeout_ms))
        if not socks:
            raise TimeoutError("No message received in given timeout")
        return self._socket.recv_pyobj()

    # --- Utilities -----------------------------------------------------
    def wait_until_ready(self, delay_s: float = 0.1, max_wait_s: float = 5.0) -> None:
        """Simple helper to wait briefly for readers to connect.

        PUSH/PULL sockets do not provide an easy "number of connected peers"
        API. This helper simply sleeps in short intervals up to `max_wait_s`.
        It is a convenience for simple single-node setups where you bind the
        writer and then spawn readers; it is NOT a robust barrier. For
        production multi-node setups prefer using torch.distributed to
        broadcast the handle and synchronize all processes.
        """
        # No-op if reader side (readers are ready when connected)
        if not self._is_writer:
            return
        waited = 0.0
        while waited < max_wait_s:
            # sleep a short interval to let readers connect
            time.sleep(delay_s)
            waited += delay_s
        # cannot reliably detect connected peers; just return after waiting
        return

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None


__all__ = ["ZmqTaskQueue", "ZmqHandle"]
