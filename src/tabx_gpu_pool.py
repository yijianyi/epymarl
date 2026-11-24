"""Cross-process GPU limits and round-robin admission for MAPPO experiment groups."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
from urllib.parse import quote

from tabx_scheduler import process_alive, process_identity
from tabx_storage import atomic_json, read_json

GROUPS = "ABCD"


class SharedGpuPool:
    def __init__(self, root, rule, gpus, capacity, group, batch_id):
        self.directory = Path(root) / "_batches" / rule / ".medium-gpu-pool"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "pool.json"
        self.config = dict(gpus=sorted(gpus), jobs_per_gpu=capacity)
        self.group = group
        self.member = dict(pid=os.getpid(), process_start=process_identity(os.getpid()), batch_id=batch_id)
        self.slots = [(gpu, self.directory / f"slot-{quote(gpu, safe='')}-{slot}")
                      for slot in range(capacity) for gpu in gpus]

    @contextmanager
    def _locked(self):
        with (self.directory / ".pool.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            state = read_json(self.path) if self.path.exists() else dict(config=self.config, groups={}, last=None)
            if state["config"] != self.config:
                raise ValueError(f"All groups using {self.directory.parent} must share --gpus and --jobs-per-gpu. "
                                 f"Pool configuration is {state['config']}")
            yield state

    def register(self):
        with self._locked() as state:
            groups = state["groups"]
            previous = groups.get(self.group, {})
            if process_alive(previous):
                raise RuntimeError(f"Group {self.group} already has a live scheduler for {previous['batch_id']}")
            groups[self.group] = dict(self.member, ready=False)
            atomic_json(self.path, state)

    def unregister(self):
        with self._locked() as state:
            previous = state["groups"].get(self.group, {})
            if all(previous.get(key) == value for key, value in self.member.items()):
                del state["groups"][self.group]
                atomic_json(self.path, state)

    def ready(self, value):
        with self._locked() as state:
            member = state["groups"][self.group]
            if member["ready"] != value:
                member["ready"] = value
                atomic_json(self.path, state)

    @staticmethod
    def _try_lock(path):
        fd = os.open(path.with_name(f"{path.name}.lock"), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return None
        except BaseException:
            os.close(fd)
            raise
        return fd

    def acquire(self, *, more_ready):
        with self._locked() as state:
            groups = state["groups"]
            start = GROUPS.index(state["last"]) + 1 if state["last"] else 0
            order = GROUPS[start:] + GROUPS[:start]
            winner = next((group for group in order if groups.get(group, {}).get("ready")
                           and process_alive(groups[group])), None)
            if winner != self.group:
                return None
            for gpu, slot in self.slots:
                fd = self._try_lock(slot)
                if fd is not None:
                    try:
                        state["last"] = self.group
                        groups[self.group]["ready"] = more_ready
                        atomic_json(self.path, state)
                    except BaseException:
                        os.close(fd)
                        raise
                    return gpu, slot, fd
        return None

    def record_launch(self, slot, job, attempt):
        atomic_json(slot.with_name(f"{slot.name}.json"), dict(batch_id=self.member["batch_id"],
                    task_id=job["task_id"], train_seed=job["train_seed"], run_id=attempt["run_id"]))

    def active_attempts(self):
        """An exec-inherited slot lock also covers a scheduler crash during spawn."""
        for _, slot in self.slots:
            path = slot.with_name(f"{slot.name}.json")
            if not path.exists():
                continue
            fd = self._try_lock(slot)
            if fd is None:
                record = read_json(path)
                if record["batch_id"] == self.member["batch_id"]:
                    yield record["run_id"]
            else:
                os.close(fd)
