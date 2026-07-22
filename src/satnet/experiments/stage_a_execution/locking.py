from __future__ import annotations

from contextlib import AbstractContextManager
import ctypes
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import socket
import time
from typing import Any, Mapping
import uuid

from .common import canonical_json_bytes, payload_hash, read_json_object
from .integrity import verify_artifact_inventory
from .ledger import read_ledger

LOCK_SCHEMA = "satnet.stage_a.execution_lock.v3"
RECOVERY_SCHEMA = "satnet.stage_a.lock_recovery_event.v3"
LOCK_DOMAIN = "satnet_stage_a_execution_lock_v3"
RECOVERY_DOMAIN = "satnet_stage_a_lock_recovery_event_v3"
CAMPAIGN_FIELDS = (
    "campaign_id", "operation", "partition", "contract_hash", "plan_hash",
    "authorization_hash", "stable_executable_commit", "tooling_proposal_hash",
)
RUN_FIELDS = ("run_key", "global_run_id", "design_id", "realization_id")


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


def host_identity() -> str:
    return f"{socket.gethostname().casefold()}:{uuid.getnode():012x}"


def campaign_identity(plan: Mapping[str, Any], authorization_hash: str) -> dict[str, Any]:
    return {
        "campaign_id": plan["campaign_manifest_hash"],
        "operation": plan["operation"],
        "partition": plan["partition"],
        "contract_hash": plan["contract_hash"],
        "plan_hash": plan["plan_hash"],
        "authorization_hash": authorization_hash,
        "stable_executable_commit": plan["stable_executable_commit"],
        "tooling_proposal_hash": plan["tooling_proposal_hash"],
    }


def run_identity(
    plan: Mapping[str, Any], authorization_hash: str, plan_run: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **campaign_identity(plan, authorization_hash),
        "run_key": plan_run["run_key"],
        "global_run_id": plan_run["global_run_id"],
        "design_id": plan_run["design_id"],
        "realization_id": plan_run["realization_id"],
    }


def lock_payload(identity: Mapping[str, Any], scope: str) -> dict[str, Any]:
    if scope not in {"campaign", "run"}:
        raise ValueError("Execution lock scope is invalid")
    pid = os.getpid()
    start = process_start_identity(pid)
    if start is None:
        raise RuntimeError("Current process start identity is unavailable")
    created_unix_ns = time.time_ns()
    payload: dict[str, Any] = {
        "schema_identifier": LOCK_SCHEMA,
        "scope": scope,
        **dict(identity),
        "host_identity": host_identity(),
        "process_id": pid,
        "process_start_identity": start,
        "creation_time": datetime.fromtimestamp(created_unix_ns / 1_000_000_000, timezone.utc).isoformat(),
        "creation_unix_ns": created_unix_ns,
        "lock_nonce": uuid.uuid4().hex,
    }
    payload["lock_hash"] = payload_hash(payload, domain=LOCK_DOMAIN)
    validate_lock_payload(payload)
    return payload


def validate_lock_payload(value: dict[str, Any]) -> None:
    claimed = value.get("lock_hash")
    payload = {field: item for field, item in value.items() if field != "lock_hash"}
    if value.get("schema_identifier") != LOCK_SCHEMA or claimed != payload_hash(payload, domain=LOCK_DOMAIN):
        raise ValueError("Execution lock identity or hash mismatch")
    scope = value.get("scope")
    required = set(CAMPAIGN_FIELDS) | {
        "schema_identifier", "scope", "host_identity", "process_id",
        "process_start_identity", "creation_time", "creation_unix_ns", "lock_nonce", "lock_hash",
    }
    if scope == "run":
        required.update(RUN_FIELDS)
    elif scope != "campaign":
        raise ValueError("Execution lock scope is invalid")
    if set(value) != required:
        raise ValueError("Execution lock field set mismatch")
    for field in CAMPAIGN_FIELDS:
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError(f"Execution lock field is invalid: {field}")
    if scope == "run":
        if type(value["global_run_id"]) is not int:
            raise ValueError("Execution lock global run ID is invalid")
        for field in ("run_key", "design_id", "realization_id"):
            if not isinstance(value[field], str) or not value[field]:
                raise ValueError(f"Execution lock field is invalid: {field}")
    if not isinstance(value["host_identity"], str) or not value["host_identity"]:
        raise ValueError("Execution lock host identity is invalid")
    if type(value["process_id"]) is not int or value["process_id"] <= 0:
        raise ValueError("Execution lock process ID is invalid")
    if not isinstance(value["process_start_identity"], str) or not value["process_start_identity"]:
        raise ValueError("Execution lock process start identity is invalid")
    if not isinstance(value["creation_time"], str) or type(value["creation_unix_ns"]) is not int:
        raise ValueError("Execution lock creation identity is invalid")
    if not isinstance(value["lock_nonce"], str) or len(value["lock_nonce"]) != 32:
        raise ValueError("Execution lock nonce is invalid")


def lock_is_stale(path: Path) -> bool:
    if not path.exists():
        return False
    value = read_json_object(path)
    validate_lock_payload(value)
    if value["host_identity"] != host_identity():
        raise PermissionError("Foreign-host execution lock cannot be classified for automatic recovery")
    observed_start = process_start_identity(value["process_id"])
    return observed_start is None or observed_start != value["process_start_identity"]


def _active_campaign_owner(campaign_root: Path, target: Path) -> bool:
    campaign_path = campaign_root.with_name(campaign_root.name + ".lock")
    if campaign_path == target or not campaign_path.exists():
        return False
    value = read_json_object(campaign_path)
    validate_lock_payload(value)
    if value["scope"] != "campaign" or value["host_identity"] != host_identity():
        raise PermissionError("Campaign owner identity cannot be proved inactive")
    return process_start_identity(value["process_id"]) == value["process_start_identity"]


def _valid_completed_output(campaign_root: Path, lock: Mapping[str, Any]) -> bool:
    if lock["operation"] == "ACCEPT":
        report_path = campaign_root / "acceptance_report.json"
        if not report_path.is_file():
            return False
        report = read_json_object(report_path)
        claimed = report.get("acceptance_report_hash")
        payload = {field: item for field, item in report.items() if field != "acceptance_report_hash"}
        return (
            report.get("acceptance_state") == "PASSED"
            and claimed == payload_hash(payload, domain="satnet_stage_a_acceptance_report_v1")
            and report.get("contract_hash") == lock["contract_hash"]
            and report.get("plan_hash") == lock["plan_hash"]
            and report.get("authorization_hash") == lock["authorization_hash"]
            and report.get("stable_executable_commit") == lock["stable_executable_commit"]
            and report.get("tooling_proposal_hash") == lock["tooling_proposal_hash"]
        )
    ledger_name = "execution_ledger.json" if lock["operation"] == "GENERATE" else "replay_ledger.json"
    ledger_path = campaign_root / ledger_name
    if not ledger_path.is_file():
        return False
    ledger = read_ledger(ledger_path)
    expected_ledger_identity = {
        "contract_hash": lock["contract_hash"],
        "plan_hash": lock["plan_hash"],
        "authorization_hash": lock["authorization_hash"],
        "stable_executable_commit": lock["stable_executable_commit"],
        "tooling_proposal_hash": lock["tooling_proposal_hash"],
        "operation": lock["operation"],
        "partition": lock["partition"],
    }
    if any(ledger.get(field) != value for field, value in expected_ledger_identity.items()):
        raise ValueError("Completed-output ledger identity differs from the recovered lock")
    records = ledger["records"]
    if lock["scope"] == "run":
        records = [record for record in records if record["run_key"] == lock["run_key"]]
        if len(records) != 1:
            raise ValueError("Recovery run identity is absent from the campaign ledger")
    completed = [record for record in records if record["state"] == "SUCCEEDED"]
    for record in completed:
        verify_artifact_inventory(campaign_root / record["output_relative_path"], record["artifacts"])
    return bool(completed) and (lock["scope"] == "run" or len(completed) == len(records))


def _write_recovery_event(root: Path, record: dict[str, Any]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{record['recovery_hash']}.json"
    encoded = canonical_json_bytes(record)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def recover_stale_lock(
    path: Path, *, expected_identity: Mapping[str, Any], minimum_age_seconds: float,
    campaign_root: Path, recovery_event_root: Path,
) -> dict[str, Any]:
    if minimum_age_seconds <= 0:
        raise ValueError("Stale-lock recovery requires a positive configured minimum age")
    if not path.is_file():
        raise FileNotFoundError(path)
    guard = path.with_name(path.name + ".recovery.lock")
    descriptor = os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    try:
        before = path.read_bytes()
        value = read_json_object(path)
        validate_lock_payload(value)
        expected = dict(expected_identity)
        identity_fields = CAMPAIGN_FIELDS + (RUN_FIELDS if value["scope"] == "run" else ())
        if {field: value[field] for field in identity_fields} != expected:
            raise PermissionError("Stale lock campaign or run identity mismatch")
        if value["host_identity"] != host_identity():
            raise PermissionError("Foreign-host execution lock cannot be automatically recovered")
        age_seconds = (time.time_ns() - value["creation_unix_ns"]) / 1_000_000_000
        if age_seconds < minimum_age_seconds:
            raise RuntimeError("Execution lock is younger than the configured recovery age")
        observed_start = process_start_identity(value["process_id"])
        if observed_start == value["process_start_identity"]:
            raise RuntimeError("Active local execution lock cannot be recovered")
        if _active_campaign_owner(campaign_root, path):
            raise RuntimeError("An active campaign owner prevents stale-lock recovery")
        if _valid_completed_output(campaign_root, value):
            raise RuntimeError("Valid completed output prevents destructive lock recovery")
        if path.read_bytes() != before:
            raise RuntimeError("Execution lock changed during stale recovery")
        recovered_unix_ns = time.time_ns()
        record: dict[str, Any] = {
            "schema_identifier": RECOVERY_SCHEMA,
            "lock_path": str(path.resolve(strict=True)),
            "recovered_lock": value,
            "minimum_age_seconds": minimum_age_seconds,
            "observed_age_seconds": age_seconds,
            "recorded_process_inactive": True,
            "observed_process_start_identity": observed_start,
            "active_campaign_owner": False,
            "valid_completed_output": False,
            "recovery_host_identity": host_identity(),
            "recovery_process_id": os.getpid(),
            "recovery_process_start_identity": process_start_identity(os.getpid()),
            "recovery_time": datetime.fromtimestamp(recovered_unix_ns / 1_000_000_000, timezone.utc).isoformat(),
            "recovered_unix_ns": recovered_unix_ns,
        }
        record["recovery_hash"] = payload_hash(record, domain=RECOVERY_DOMAIN)
        event_path = _write_recovery_event(recovery_event_root, record)
        path.unlink()
        record["recovery_event_path"] = str(event_path.resolve(strict=True))
        return record
    finally:
        guard.unlink(missing_ok=True)


class ExclusiveLock(AbstractContextManager["ExclusiveLock"]):
    def __init__(self, path: Path, identity: Mapping[str, Any], scope: str = "campaign") -> None:
        self.path = path
        self.identity = identity
        self.scope = scope
        self.acquired = False
        self.payload: dict[str, Any] | None = None

    def acquire(self) -> "ExclusiveLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = lock_payload(self.identity, self.scope)
        encoded = canonical_json_bytes(payload)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            value = read_json_object(self.path)
            validate_lock_payload(value)
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


def campaign_lock(
    root: Path, plan: Mapping[str, Any], authorization_hash: str,
) -> ExclusiveLock:
    return ExclusiveLock(
        root.with_name(root.name + ".lock"), campaign_identity(plan, authorization_hash), "campaign",
    )


def per_run_lock(
    campaign_root: Path, plan: Mapping[str, Any], authorization_hash: str,
    plan_run: Mapping[str, Any],
) -> ExclusiveLock:
    return ExclusiveLock(
        campaign_root / "operational" / "locks" / f"{plan_run['run_key']}.lock",
        run_identity(plan, authorization_hash, plan_run), "run",
    )
