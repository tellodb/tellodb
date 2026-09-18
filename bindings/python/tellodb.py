"""Python binding for tellodb, over the C ABI in crates/tellodb-ffi.

Only the standard library is needed: ctypes loads the shared library built by
`cargo build --release -p tellodb-ffi`.

    from tellodb import Tellodb

    with Tellodb("./agent-memory") as db:
        db.remember("alice", "I just moved to Denver.", session_id="chat-1", turn_index=0)
        for hit in db.recall("where do I live?", entity_id="alice", limit=5):
            print(hit["score"], hit["text"])

Set TELLODB_LIB to the library path, or pass `library=` to the constructor.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
from pathlib import Path
from typing import Any, Iterable

__all__ = ["Tellodb", "TellodbError"]


class TellodbError(RuntimeError):
    """An error reported by the engine."""


def _library_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "libtellodb.dylib"
    if system == "Windows":
        return "tellodb.dll"
    return "libtellodb.so"


def _find_library() -> str:
    from_env = os.environ.get("TELLODB_LIB")
    if from_env:
        return from_env
    name = _library_name()
    # Installed alongside this file, then the repository's build outputs.
    here = Path(__file__).resolve().parent
    repo = here.parents[2] if len(here.parents) > 2 else here
    candidates = [
        here / name,
        *(repo / "target" / profile / name for profile in ("release", "fastrelease", "debug")),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise TellodbError(
        f"cannot find {name}; build it with `cargo build --release -p tellodb-ffi` "
        f"or set TELLODB_LIB. Looked in: " + ", ".join(str(c) for c in candidates)
    )


def _bind(library: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(library)
    lib.tellodb_open.argtypes = [ctypes.c_char_p]
    lib.tellodb_open.restype = ctypes.c_void_p
    for name in ("tellodb_ingest", "tellodb_query"):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        fn.restype = ctypes.c_void_p
    lib.tellodb_current_fact.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.tellodb_current_fact.restype = ctypes.c_void_p
    lib.tellodb_last_error.argtypes = []
    lib.tellodb_last_error.restype = ctypes.c_char_p
    lib.tellodb_string_free.argtypes = [ctypes.c_void_p]
    lib.tellodb_string_free.restype = None
    lib.tellodb_close.argtypes = [ctypes.c_void_p]
    lib.tellodb_close.restype = None
    return lib


class Tellodb:
    """An open tellodb data directory."""

    def __init__(self, data_dir: str | os.PathLike[str], library: str | None = None) -> None:
        self._lib = _bind(library or _find_library())
        handle = self._lib.tellodb_open(str(data_dir).encode())
        if not handle:
            raise TellodbError(self._last_error() or "failed to open database")
        self._handle = ctypes.c_void_p(handle)

    def _last_error(self) -> str | None:
        message = self._lib.tellodb_last_error()
        return message.decode(errors="replace") if message else None

    def _take_json(self, pointer: int | None) -> Any:
        if not pointer:
            raise TellodbError(self._last_error() or "call failed")
        try:
            return json.loads(ctypes.cast(pointer, ctypes.c_char_p).value or b"null")
        finally:
            self._lib.tellodb_string_free(ctypes.c_void_p(pointer))

    def _require_open(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise TellodbError("database is closed")
        return self._handle

    def ingest(self, memories: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Stores memories; each needs `entity_id` and `text`."""
        payload = json.dumps(list(memories)).encode()
        return self._take_json(self._lib.tellodb_ingest(self._require_open(), payload))

    def remember(
        self,
        entity_id: str,
        text: str,
        *,
        session_id: str | None = None,
        turn_index: int | None = None,
        role: str | None = None,
        timestamp_ms: int | None = None,
        kind: str | None = None,
        memory_id: str | None = None,
    ) -> dict[str, Any]:
        """Stores one memory."""
        memory = {
            "entity_id": entity_id,
            "text": text,
            "session_id": session_id,
            "turn_index": turn_index,
            "role": role,
            "timestamp_ms": timestamp_ms,
            "kind": kind,
            "memory_id": memory_id,
        }
        return self.ingest([{k: v for k, v in memory.items() if v is not None}])

    def recall(
        self,
        text: str,
        *,
        entity_id: str | None = None,
        limit: int = 10,
        as_of_ms: int | None = None,
        reference_time_ms: int | None = None,
        rerank: bool = False,
    ) -> list[dict[str, Any]]:
        """Searches memories, best first."""
        query = {
            "text": text,
            "entity_id": entity_id,
            "limit": limit,
            "as_of_ms": as_of_ms,
            "reference_time_ms": reference_time_ms,
            "rerank": rerank,
        }
        payload = json.dumps({k: v for k, v in query.items() if v is not None}).encode()
        return self._take_json(self._lib.tellodb_query(self._require_open(), payload))

    def current_fact(self, entity_id: str, fact_key: str) -> str | None:
        """The current value of a tracked fact, or None."""
        result = self._take_json(
            self._lib.tellodb_current_fact(
                self._require_open(), entity_id.encode(), fact_key.encode()
            )
        )
        return result.get("value")

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self._lib.tellodb_close(self._handle)
            self._handle = None

    def __enter__(self) -> "Tellodb":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
