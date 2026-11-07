"""Round-robin admission from durable per-run registrations to bounded ready work."""
from collections import Counter, deque
import json

from tabx_reporting import LineCursor


class EvalDispatcher:
    def __init__(self, queue, rule, batch_id, workers, prefetch=4, on_request=None, known_status=None):
        import fcntl
        if workers < 1 or prefetch < 1:
            raise ValueError("Evaluation workers and prefetch must be positive")
        self.queue, self.rule, self.batch_id = queue, rule, batch_id
        self.directory = queue.directory(rule, batch_id)
        self.lock = (self.directory / ".dispatcher.lock").open("a")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.lock.close()
            raise RuntimeError("Another dispatcher owns this evaluation queue") from error
        queue.configure_prefetch(rule, batch_id, workers, prefetch)
        self.limit = workers * prefetch
        self.on_request, self.known_status = on_request, known_status
        self.cursors, self.requests, self.statuses, self.backlogs = {}, {}, {}, {}
        self.ready_runs, self.run_set, self.queued = deque(), set(), set()
        self.counts = Counter()
        self.last_admission = None

    def observe(self, key, status):
        previous = self.statuses.get(key)
        if previous == status:
            return
        if previous is not None:
            self.counts[previous] -= 1
        self.statuses[key] = status
        self.counts[status] += 1
        if status != "registered":
            self.queued.discard(key)

    def _ingest(self, descriptor):
        run = descriptor["run_key"]
        cursor = self.cursors.get(run)
        if cursor is None:
            cursor = LineCursor(self.queue.root / descriptor["ledger"])
            self.cursors[run] = cursor
        for line in cursor.read():
            entry = json.loads(line)
            key, request = entry["request_id"], entry["request"]
            if key in self.queued:
                continue
            cached = self.known_status(key) if self.known_status else None
            if cached == "complete" or self.statuses.get(key) == "complete":
                self.observe(key, "complete")
                continue
            state = self.queue.status(key)
            if state is None:
                raise ValueError("Registered request lost its durable queue state")
            status = state["status"]
            self.requests[key] = request
            changed = self.statuses.get(key) != status
            self.observe(key, status)
            if changed and self.on_request:
                self.on_request(request, status)
            if status == "registered":
                self.backlogs.setdefault(run, deque()).append(key)
                self.queued.add(key)
                if run not in self.run_set:
                    self.ready_runs.append(run)
                    self.run_set.add(run)

    def restore(self):
        # Startup only: normalize ready work after a worker-count/prefetch change.
        pending = [key for key, state in self.queue.active_keys(self.rule, self.batch_id).items() if state == "pending"]
        for key in pending[self.limit:]:
            self.queue.stage(key)
        for path in (self.directory / "runs").glob("*.json"):
            self._ingest(self.queue._read(path))

    def poll(self):
        for notice in self.queue.registration_updates(self.rule, self.batch_id):
            self._ingest(notice)
            self.queue.acknowledge_registration(notice)
        active = self.queue.active_keys(self.rule, self.batch_id)
        available = self.limit - sum(status == "pending" for status in active.values())
        while available > 0 and self.ready_runs:
            run = self.ready_runs.popleft()
            self.run_set.remove(run)
            backlog = self.backlogs[run]
            key = backlog.popleft()
            if key in self.queued:
                self.queued.remove(key)
                status = self.queue.expose(key)
                self.observe(key, status)
                if self.on_request:
                    self.on_request(self.requests[key], status)
                if status == "pending":
                    available -= 1
            if backlog:
                self.ready_runs.append(run)
                self.run_set.add(run)
            else:
                del self.backlogs[run]
        admission = dict(registered_waiting=self.counts["registered"], ready_limit=self.limit)
        if admission != self.last_admission:
            self.queue._write(self.directory / "admission.json", admission)
            self.last_admission = admission
        return active

    def close(self):
        self.lock.close()
