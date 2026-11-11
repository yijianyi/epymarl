"""Atomic file queue, with immutable per-seed results and consumable notifications."""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import time
import uuid


class EvalQueue:
    def __init__(self, root):
        from tabx_storage import atomic_json, read_json, resolve_under, safe_component
        self.root = Path(root).expanduser().resolve()
        self.index = self.root / "_eval_requests"
        (self.index / "locks").mkdir(parents=True, exist_ok=True)
        self._write, self._read = atomic_json, read_json
        self._resolve, self._safe = resolve_under, safe_component
        self._info, self._directories, self._last_reclaim, self._policies = {}, {}, {}, {}

    @staticmethod
    def request_id(payload):
        # Keep the existing algorithm AND payload unchanged.
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def directory(self, rule, batch_id=None):
        if rule not in ("hp", "loss"):
            raise ValueError("Unknown evaluation rule")
        scope = (rule, batch_id)
        if scope not in self._directories:
            path = (self.root / "_batches" / rule / self._safe(batch_id) / "eval_queue"
                    if batch_id is not None else self.root / "_eval_queue" / rule)
            for name in ("pending", "running", "finished", "failed", "archive/complete", "archive/failed",
                         "registrations", "runs", "registration_locks"):
                (path / name).mkdir(parents=True, exist_ok=True)
            self._directories[scope] = path
        return self._directories[scope]

    def configure_prefetch(self, rule, batch_id, workers, prefetch):
        value = dict(workers=workers, prefetch=prefetch, ready_limit=workers * prefetch)
        self._write(self.directory(rule, batch_id) / "prefetch.json", value)
        self._policies[(rule, batch_id)] = value

    def policy(self, rule, batch_id=None):
        scope = (rule, batch_id)
        if scope not in self._policies:
            path = self.directory(rule, batch_id) / "prefetch.json"
            self._policies[scope] = self._read(path) if path.exists() else None
        return self._policies[scope]

    def _initial_status(self, request):
        return "registered" if self.policy(request["rule"], request.get("batch_id")) else "pending"

    def _registered_path(self, info):
        request = info["request"]
        parent = (self._resolve(self.root, request["run"]) / "evaluation_requests" if request.get("run")
                  else self._resolve(self.root, info["directory"]))
        return parent / "registered" / f"{info['request_id']}.json"

    @contextmanager
    def _registration_lock(self, directory, run_key):
        import fcntl
        with (directory / "registration_locks" / f"{run_key}.lock").open("a") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            yield

    def _ensure_ledger(self, info, force=False):
        request = info["request"]
        if not self.policy(request["rule"], request.get("batch_id")):
            return
        if not force and info.get("ledger_registered"):
            return
        directory = self._resolve(self.root, info["directory"])
        run = request.get("run", f"_queue_only/{info['request_id']}")
        run_key = hashlib.sha256(run.encode()).hexdigest()[:32]
        ledger = (self._resolve(self.root, run) / "evaluation_requests" / "requests.jsonl"
                  if request.get("run") else directory / "queue_only" / run_key / "requests.jsonl")
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with self._registration_lock(directory, run_key):
            descriptor = dict(run_key=run_key, run_id=request.get("run_id", run_key), run=run,
                              directory=str(directory.relative_to(self.root)),
                              ledger=str(ledger.relative_to(self.root)))
            target = directory / "runs" / f"{run_key}.json"
            if not target.exists():
                self._write(target, descriptor)
            line = json.dumps(dict(request_id=info["request_id"], request=request,
                                   checkpoint_step=info["checkpoint_step"]), separators=(",", ":")).encode() + b"\n"
            with ledger.open("ab") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
                end = stream.tell()
            self._write(directory / "registrations" / f"{run_key}.json",
                        dict(descriptor, end=end, version=uuid.uuid4().hex))
            info["ledger_registered"] = True
            self._write(self.index / f"{info['request_id']}.json", info)

    def registration_updates(self, rule, batch_id):
        return [self._read(path) for path in (self.directory(rule, batch_id) / "registrations").glob("*.json")]

    def acknowledge_registration(self, notice):
        directory = self.directory_for_notice(notice)
        path = directory / "registrations" / f"{notice['run_key']}.json"
        with self._registration_lock(directory, notice["run_key"]):
            if path.exists() and self._read(path)["version"] == notice["version"]:
                path.unlink()

    def directory_for_notice(self, notice):
        # Registration descriptors carry their queue scope independently of payload.
        return self._resolve(self.root, notice["directory"])

    @contextmanager
    def _lock(self, key):
        import fcntl
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("Invalid evaluation request ID")
        with (self.index / "locks" / f"{key}.lock").open("a") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            yield

    def info(self, key):
        if key not in self._info:
            path = self.index / f"{key}.json"
            if not path.exists():
                return None
            info = self._read(path)
            if info["request_id"] != key or self.request_id(info["request"]) != key:
                raise ValueError("Evaluation request index identity mismatch")
            self._info[key] = info
        return self._info[key]

    def _register(self, payload):
        key = self.request_id(payload)
        info = self.info(key)
        if info is None:
            directory = self.directory(payload["rule"], payload.get("batch_id"))
            snapshot = Path(payload.get("snapshot", ""))
            info = dict(request_id=key, request=payload, directory=str(directory.relative_to(self.root)),
                        checkpoint_step=int(snapshot.name) if snapshot.name.isdigit() else None)
            self._write(self.index / f"{key}.json", info)
            self._info[key] = info
        elif info["request"] != payload:
            raise ValueError("Request ID collision")
        return info

    def _find(self, key):
        info = self.info(key)
        if info is None:
            return None, None
        directory = self._resolve(self.root, info["directory"])
        for folder, status in (("pending", "pending"), ("running", "running"), ("finished", "complete"),
                               ("failed", "failed"), ("archive/complete", "complete"), ("archive/failed", "failed")):
            path = directory / folder / f"{key}.json"
            if path.exists():
                record = self._read(path)
                record["status"] = status
                return path, record
        path = self._registered_path(info)
        if path.exists():
            record = self._read(path)
            record["status"] = "registered"
            return path, record
        return None, None

    def _new_record(self, info):
        return dict(info, status="pending", owner=None, updated=time.time(), attempts=0, error=None,
                    version=uuid.uuid4().hex)

    def _transition(self, path, record, status):
        directory = self._resolve(self.root, record["directory"])
        folder = "finished" if status == "complete" else status
        destination = (self._registered_path(record) if status == "registered"
                       else directory / folder / f"{record['request_id']}.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        record.update(status=status, updated=time.time(), version=uuid.uuid4().hex)
        if path is not None and path != destination:
            os.replace(path, destination)
        self._write(destination, record)
        return record

    def _evaluation_dir(self, info):
        request = info["request"]
        if request.get("run"):
            return self._resolve(self.root, request["run"]) / "evaluations"
        # Small queue-only callers need not construct a training run.
        return self._resolve(self.root, info["directory"]) / "results"

    def result_path(self, key):
        info = self.info(key)
        if info is None:
            raise KeyError(key)
        return self._evaluation_dir(info) / f"{key}.json"

    def _validate_report(self, info, report):
        request = info["request"]
        if report.get("request") != request:
            raise ValueError("Evaluation result/request identity mismatch")
        from tabx_difficulty import evaluation_difficulty
        evaluation_difficulty(report, request_id=info.get("request_id"))
        rows = report.get("episodes", [])
        seeds = [row["seed"] for row in rows]
        if (len(seeds) != len(request["seeds"]) or len(set(seeds)) != len(seeds)
                or set(seeds) != set(request["seeds"])):
            raise ValueError("Evaluation result is missing or repeats requested seeds")
        for row in rows:
            if "return_value" in row and not math.isfinite(float(row["return_value"])):
                raise ValueError("Non-finite evaluation return")
        if request.get("snapshot"):
            manifest = self._read(self._resolve(self.root, request["snapshot"]) / "manifest.json")
            for field, source in (("checkpoint_hash", "agent_sha256"), ("run_id", "run_id"),
                                  ("task_id", "task_id"), ("task_hash", "task_hash"),
                                  ("bank_hash", "bank_hash"), ("rule", "timeout_rule")):
                if field in request and request[field] != manifest.get(source):
                    raise ValueError(f"Evaluation result checkpoint mismatch: {field}")
            if manifest["t_env"] != info["checkpoint_step"]:
                raise ValueError("Evaluation result checkpoint step mismatch")
        return report

    def completed_result(self, key):
        info = self.info(key)
        if info is None or not self.result_path(key).exists():
            return None
        return self._validate_report(info, self._read(self.result_path(key)))

    def _summary(self, info, report):
        if info["request"].get("run"):
            from tabx_reporting import update_eval_summary
            update_eval_summary(self._resolve(self.root, info["request"]["run"]),
                                info["request_id"], report, info["checkpoint_step"])

    def enqueue(self, payload, max_pending=None):
        """Register once; max_pending is retained as a compatibility argument."""
        key = self.request_id(payload)
        with self._lock(key):
            info = self._register(payload)
            path, record = self._find(key)
            if record is not None:
                self._ensure_ledger(info)
                return key
            record = self._new_record(info)
            report = self.completed_result(key)
            if report is not None:
                self._summary(info, report)
                self._transition(None, record, "complete")
            else:
                self._transition(None, record, self._initial_status(payload))
            self._ensure_ledger(info)
        return key

    def recover(self, payload, retry_failed=False, *, defer_summary=False):
        """One-time reconciliation at training completion/resume, never an idle poll."""
        key = self.request_id(payload)
        with self._lock(key):
            info = self._register(payload)
            path, record = self._find(key)
            record = record or self._new_record(info)
            try:
                report = self.completed_result(key)
            except (ValueError, KeyError, OSError) as error:
                if retry_failed:
                    bad = self.result_path(key)
                    preserved = bad.parent / "invalid" / f"{key}-{uuid.uuid4().hex}.json"
                    preserved.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(bad, preserved)
                    report = None
                    record["status"] = "failed"
                else:
                    record["error"] = str(error)
                    return dict(self._transition(path, record, "failed"), report=None)
            if report is not None:
                if not defer_summary:
                    self._summary(info, report)
                if record["status"] != "complete" or path is None:
                    record = self._transition(path, record, "complete")
                self._ensure_ledger(info)
                return dict(record, report=report)
            rescheduled = False
            if path is None or record["status"] == "complete" or (retry_failed and record["status"] == "failed"):
                rescheduled = path is not None
                record.update(owner=None, error=None)
                record = self._transition(path, record, self._initial_status(payload))
            self._ensure_ledger(info, force=rescheduled and record["status"] == "registered")
            return dict(record, report=None)

    def expose(self, key):
        with self._lock(key):
            path, record = self._find(key)
            if record is None:
                raise KeyError(key)
            if record["status"] == "registered":
                self._transition(path, record, "pending")
            return record["status"]

    def stage(self, key):
        """Return ready work to registration during bounded-queue reconstruction."""
        with self._lock(key):
            path, record = self._find(key)
            if record and record["status"] == "pending":
                self._transition(path, record, "registered")
                self._ensure_ledger(self.info(key), force=True)

    def active_keys(self, rule, batch_id=None):
        """Read directory entries only: no old requests, manifests or result files."""
        directory = self.directory(rule, batch_id)
        return {path.stem: status for status in ("pending", "running")
                for path in (directory / status).glob("*.json")}

    def _reclaim_expired(self, rule, batch_id, lease_seconds):
        scope = (rule, batch_id)
        now = time.monotonic()
        if now - self._last_reclaim.get(scope, float("-inf")) < min(30, max(lease_seconds, 0)):
            return
        self._last_reclaim[scope] = now
        for candidate in (self.directory(rule, batch_id) / "running").glob("*.json"):
            with self._lock(candidate.stem):
                path, record = self._find(candidate.stem)
                if record and record["status"] == "running" and record["updated"] < time.time() - lease_seconds:
                    record["owner"] = None
                    target = self._initial_status(record["request"])
                    self._transition(path, record, target)
                    if target == "registered":
                        self._ensure_ledger(self.info(candidate.stem), force=True)

    def claim(self, rule, lease_seconds=3600, batch_id=None, request_id=None):
        directory = self.directory(rule, batch_id)
        self._reclaim_expired(rule, batch_id, lease_seconds)
        candidates = ([directory / "pending" / f"{request_id}.json"] if request_id else
                      (directory / "pending").glob("*.json"))
        for candidate in candidates:
            with self._lock(candidate.stem):
                if not candidate.exists():
                    continue
                record = self._read(candidate)
                owner = uuid.uuid4().hex
                record.update(owner=owner, worker_pid=os.getpid(), worker_host=socket.gethostname(),
                              attempts=record["attempts"] + 1, error=None)
                # The move is atomic; another worker cannot claim this request.
                self._transition(candidate, record, "running")
                return candidate.stem, owner, record["request"]
        return None

    def _owned(self, key, owner):
        path, record = self._find(key)
        if record is None or record["status"] != "running" or record["owner"] != owner:
            raise RuntimeError("Evaluation lease lost")
        return path, record

    def heartbeat(self, key, owner):
        with self._lock(key):
            path, record = self._owned(key, owner)
            record["updated"] = time.time()
            self._write(path, record)

    def record(self, key, owner, result):
        with self._lock(key):
            path, record = self._owned(key, owner)
            seed = int(result["seed"])
            if seed not in record["request"]["seeds"]:
                raise ValueError("Unexpected evaluation seed")
            target = self._evaluation_dir(record) / "partial" / key / f"{seed}.json"
            if not target.exists():
                self._write(target, result)
            record["updated"] = time.time()
            self._write(path, record)

    def results(self, key):
        report = self.completed_result(key)
        if report is not None:
            return sorted(report["episodes"], key=lambda row: row["seed"])
        info = self.info(key)
        if info is None:
            return []
        rows = []
        for seed in sorted(info["request"]["seeds"]):
            path = self._evaluation_dir(info) / "partial" / key / f"{seed}.json"
            if path.exists():
                row = self._read(path)
                if row["seed"] != seed:
                    raise ValueError("Partial evaluation seed mismatch")
                rows.append(row)
        return rows

    def publish_result(self, key, owner, report):
        with self._lock(key):
            _, record = self._owned(key, owner)
            self._validate_report(record, report)
            self._write(self.result_path(key), report)

    def finish(self, key, owner, error=None):
        with self._lock(key):
            path, record = self._owned(key, owner)
            if error is None:
                if not record["request"].get("run") and not self.result_path(key).exists():
                    self._write(self.result_path(key), dict(request=record["request"], episodes=self.results(key)))
                report = self.completed_result(key)
                if report is None:
                    raise ValueError("Publish the complete evaluation JSON before its notification")
                self._summary(record, report)
            record["error"] = error
            self._transition(path, record, "failed" if error else "complete")

    def status(self, key):
        with self._lock(key):
            _, record = self._find(key)
            return ({field: record.get(field) for field in ("status", "error", "owner", "updated", "attempts", "version")}
                    if record else None)

    def statuses(self, keys):
        return {key: dict(id=key, **status) for key in set(keys) if (status := self.status(key)) is not None}

    def notifications(self, rule, batch_id=None):
        directory = self.directory(rule, batch_id)
        records = []
        for folder in ("finished", "failed"):
            for candidate in (directory / folder).glob("*.json"):
                with self._lock(candidate.stem):
                    if candidate.exists():
                        record = self._read(candidate)
                        record["status"] = "complete" if folder == "finished" else "failed"
                        records.append(record)
        return records

    def acknowledge(self, record):
        key = record["request_id"]
        with self._lock(key):
            path, current = self._find(key)
            if current is None or current["version"] != record["version"]:
                return
            if path.parent.name not in ("finished", "failed"):
                return
            destination = self._resolve(self.root, current["directory"]) / "archive" / current["status"] / path.name
            os.replace(path, destination)

    def invalidate(self, key, error):
        with self._lock(key):
            path, record = self._find(key)
            if record is None:
                raise KeyError(key)
            record["error"] = error
            return self._transition(path, record, "failed")

    def retry_failed(self, rule, batch_id=None, *, defer_summary=False):
        directory = self.directory(rule, batch_id)
        records = [self._read(path) for folder in ("failed", "archive/failed")
                   for path in (directory / folder).glob("*.json")]
        for record in records:
            self.recover(record["request"], retry_failed=True, defer_summary=defer_summary)
        return len(records)

    def release_batch(self, rule, batch_id):
        """Only call after the owning batch's worker process group has stopped."""
        if not batch_id:
            raise ValueError("Releasing workers requires an explicit batch_id")
        count = 0
        for candidate in (self.directory(rule, batch_id) / "running").glob("*.json"):
            with self._lock(candidate.stem):
                path, record = self._find(candidate.stem)
                if record and record["status"] == "running":
                    record["owner"] = None
                    target = self._initial_status(record["request"])
                    self._transition(path, record, target)
                    if target == "registered":
                        self._ensure_ledger(self.info(candidate.stem), force=True)
                    count += 1
        return count
