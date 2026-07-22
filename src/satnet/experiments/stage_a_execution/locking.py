from __future__ import annotations

from contextlib import AbstractContextManager
import ctypes
import json
import os
from pathlib import Path
import platform
import time
from typing import Any
import uuid

from .common import atomic_write_json, payload_hash, read_json_object

LOCK_SCHEMA = "satnet.stage_a.execution_lock.v2"
RECOVERY_SCHEMA = "satnet.stage_a.lock_recovery.v2"


def _windows_process_start(pid: int) -> str | None:
    process_query_limited_information = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return None
    creation = ctypes.c_ulonglong()
    exit_time = ctypes.c_ulonglong()
    kernel = ctypes.c_ulonglong()
    user = ctypes.c_ulonglong()
    try:
        success = ctypes.windll.kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        )
        return str(creation.value) if success else None
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def process_start_identity(pid: int) -> str | None:
    if type(pid) is not int or pid <= 0:
        return None
    if platform.system() == "Windows":
        return _windows_process_start(pid)
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        return stat_path.read_text(encoding="utf-8").split()[21]
    except (FileNotFoundError, IndexError, OSError):
        try:
            os.kill(pid, 0)
        except OSError:
            return None
        return "alive-start-unavailable"


def lock_payload(identity: str, scope: str) -> dict[str, Any]:
    pid = os.getpid()
    start = process_start_identity(pid)
    if start is None:
        raise RuntimeError("Current process start identity is unavailable")
    payload: dict[str, Any] = {
        "schema_identifier": LOCK_SCHEMA,
        "identity": identity,
        "scope": scope,
        "pid": pid,
        "process_start_identity": start,
        "lock_token": uuid.uuid4().hex,
        "created_unix_ns": time.time_ns(),
    }
    payload["lock_hash"] = payload_hash(payload, domain="satnet_stage_a_execution_lock_v2")
    return payload


def validate_lock_payload(value: dict[str, Any]) -> None:
    claimed = value.get("lock_hash")
    payload = {field: item for field, item in value.items() if field != "lock_hash"}
    if value.get("schema_identifier") != LOCK_SCHEMA or claimed != payload_hash(payload, domain="satnet_stage_a_execution_lock_v2"):
        raise ValueError("Execution lock identity or hash mismatch")
    if not isinstance(value.get("identity"), str) or not value["identity"]:
        raise ValueError("Execution lock identity is invalid")
    if not isinstance(value.get("scope"), str) or not value["scope"]:
        raise ValueError("Execution lock scope is invalid")
    if not isinstance(value.get("lock_token"), str) or len(value["lock_token"]) != 32:
        raise ValueError("Execution lock token is invalid")
    if type(value.get("pid")) is not int or value["pid"] <= 0:
        raise ValueError("Execution lock process ID is invalid")
    if not isinstance(value.get("process_start_identity"), str) or not value["process_start_identity"]:
        raise ValueError("Execution lock process start identity is invalid")


def lock_is_stale(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        value = read_json_object(path)
        validate_lock_payload(value)
    except (OSError, ValueError, json.JSONDecodeError):
        return True
    observed_start = process_start_identity(value["pid"])
    return observed_start is None or observed_start != value["process_start_identity"]


def recover_stale_lock(path: Path, *, expected_identity: str, recovery_log: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    recovery_guard = path.with_name(path.name + ".recovery.lock")
    with ExclusiveLock(recovery_guard, expected_identity, "stale-lock-recovery"):
        value = read_json_object(path)
        validate_lock_payload(value)
        if value["identity"] != expected_identity:
            raise PermissionError("Stale lock identity differs from authorized recovery identity")
        if not lock_is_stale(path):
            raise RuntimeError("Live execution lock cannot be recovered")
        record: dict[str, Any] = {
            "schema_identifier": RECOVERY_SCHEMA,
            "lock_path": str(path.resolve(strict=True)),
            "recovered_lock": value,
            "recovery_pid": os.getpid(),
            "recovery_process_start_identity": process_start_identity(os.getpid()),
            "recovered_unix_ns": time.time_ns(),
        }
        record["recovery_hash"] = payload_hash(record, domain="satnet_stage_a_lock_recovery_v2")
        recovery_log.parent.mkdir(parents=True, exist_ok=True)
        if recovery_log.exists():
            existing = read_json_object(recovery_log)
            records = existing.get("records")
            if not isinstance(records, list):
                raise ValueError("Lock recovery log is malformed")
            records.append(record)
            existing["log_hash"] = payload_hash({"records": records}, domain="satnet_stage_a_lock_recovery_log_v2")
            atomic_write_json(recovery_log, existing, overwrite=True)
        else:
            log = {"schema_identifier": "satnet.stage_a.lock_recovery_log.v2", "records": [record]}
            log["log_hash"] = payload_hash({"records": log["records"]}, domain="satnet_stage_a_lock_recovery_log_v2")
            atomic_write_json(recovery_log, log)
        current = read_json_object(path)
        if current != value:
            raise RuntimeError("Execution lock changed during stale recovery")
        path.unlink()
        return record


class ExclusiveLock(AbstractContextManager["ExclusiveLock"]):
    def __init__(self, path: Path, identity: str, scope: str = "campaign") -> None:
        self.path = path
        self.identity = identity
        self.scope = scope
        self.acquired = False
        self.payload: dict[str, Any] | None = None

    def acquire(self) -> "ExclusiveLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = lock_payload(self.identity, self.scope)
        encoded = (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            state = "stale" if lock_is_stale(self.path) else "live"
            raise RuntimeError(f"Execution lock already exists and is {state}; explicit validated recovery is required: {self.path}") from error
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            self.path.unlink(missing_ok=True)
            raise
        self.payload = payload
        self.acquired = True
        return self

    def __enter__(self) -> "ExclusiveLock":
        return self.acquire()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.acquired:
            current = read_json_object(self.path)
            if current != self.payload:
                raise RuntimeError("Execution lock ownership changed before release")
            self.path.unlink(missing_ok=False)
            self.acquired = False


def campaign_lock(root: Path, identity: str) -> ExclusiveLock:
    return ExclusiveLock(root.with_name(root.name + ".lock"), identity, "campaign")


def per_run_lock(campaign_root: Path, run_key: str, identity: str) -> ExclusiveLock:
    return ExclusiveLock(campaign_root / "operational" / "locks" / f"{run_key}.lock", identity, f"run:{run_key}")
