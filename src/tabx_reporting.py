"""CSV/JSON summaries for TABX. No plotting or optional graphics dependencies."""
from contextlib import contextmanager
import csv
import io
import json
import os
from pathlib import Path
import statistics
import tempfile

TRAIN_METRICS_VERSION = "tabx-train-metrics-v3-td-bc-total"

TRAIN_FIELDS = ("t_env", "learner_updates", "loss", "td_error_abs", "grad_norm_before_clip",
                "grad_norm_clip", "q_taken_mean", "target_mean", "lr", "bc_loss", "bc_weight", "total_loss")
EVAL_FIELDS = ("request_id", "checkpoint_step", "phase", "action_selection", "episodes", "wins",
               "win_rate_pct", "mean_return", "return_std", "elimination_wins", "timeout_wins", "pure_timeouts")


def append_train_metrics(path, row):
    """Only the owning trainer writes this file; each row uses the update's t_env."""
    path = Path(path)
    # Never append the new meaning/columns to a historical metrics file.
    if path.exists() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8") as stream:
            if next(csv.reader(stream)) != list(TRAIN_FIELDS):
                raise ValueError("Training metrics schema differs; start a new run and preserve the historical CSV")
    row = dict(row)
    row.setdefault("bc_loss", None)
    row.setdefault("bc_weight", 0.0)
    row.setdefault("total_loss", row["loss"])
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRAIN_FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()


def write_csv(path, fields, rows):
    """Publish a whole summary atomically so live readers never see half a CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".summary-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _complete_text(path):
    try:
        data = Path(path).read_bytes()
    except FileNotFoundError:
        return ""
    # A live append-only producer may be in the middle of its last line.
    return data[:data.rfind(b"\n") + 1].decode("utf-8")


def read_csv(path):
    return list(csv.DictReader(io.StringIO(_complete_text(path))))


def read_episodes(path):
    return [json.loads(line) for line in _complete_text(path).splitlines() if line.strip()]


class LineCursor:
    """Byte cursor that consumes complete lines only, including across rotation."""
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.identity = None
        self.reset = False

    def read(self):
        self.reset = False
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return []
        identity = (stat.st_dev, stat.st_ino)
        if identity != self.identity or stat.st_size < self.offset:
            self.offset, self.identity, self.reset = 0, identity, True
        if stat.st_size == self.offset:
            return []
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            data = stream.read()
        end = data.rfind(b"\n") + 1
        self.offset += end
        return data[:end].decode("utf-8").splitlines()


class TrainingLogReader:
    """Only active runs retain readers. Returned metric rows are newly appended."""
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.metrics = LineCursor(self.run_dir / "train_metrics.csv")
        self.episodes = LineCursor(self.run_dir / "training_episodes.jsonl")
        self.header = None
        self.summary_stamp = None

    def update(self, view):
        lines = self.metrics.read()
        if self.metrics.reset:
            self.header = None
        rows = []
        for line in lines:
            values = next(csv.reader([line]))
            if self.header is None:
                self.header = values
            else:
                row = dict(zip(self.header, values))
                rows.append(row)
                view.update(t_env=max(view.get("t_env", 0), int(row["t_env"])),
                            learner_updates=max(view.get("learner_updates", 0), int(row["learner_updates"])),
                            latest_metric=row)
        episodes = self.episodes.read()
        if episodes:
            episode = json.loads(episodes[-1])
            from envs.tabx_outcome import END_REASONS
            view.update(t_env=max(view.get("t_env", 0), int(episode["t_env"])),
                        last_end_reason=END_REASONS.get(int(episode["end_reason"])))
        summary = self.run_dir / "training_summary.json"
        try:
            stamp = (summary.stat().st_mtime_ns, summary.stat().st_size)
        except FileNotFoundError:
            stamp = None
        changed = bool(rows or episodes)
        if stamp is not None and stamp != self.summary_stamp:
            from tabx_storage import read_json
            view.update(read_json(summary))
            self.summary_stamp = stamp
            changed = True
        return rows, changed


@contextmanager
def summary_lock(run_dir):
    # CPU workers may complete different checkpoints of one run concurrently.
    import fcntl
    with (Path(run_dir) / ".eval-summary.lock").open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def evaluation_summary(request_id, report, checkpoint_step):
    rows = report["episodes"]
    if not rows or len(rows) != len(report["request"]["seeds"]):
        raise ValueError("Evaluation summary requires every requested episode")
    if len({row["seed"] for row in rows}) != len(rows) or {row["seed"] for row in rows} != set(report["request"]["seeds"]):
        raise ValueError("Evaluation summary seeds do not match the request")
    returns = [float(row["return_value"]) for row in rows]
    wins = sum(bool(row["is_win"]) for row in rows)
    return dict(request_id=request_id, checkpoint_step=int(checkpoint_step), phase=report["request"]["phase"],
                action_selection=report["request"]["action_selection"], episodes=len(rows), wins=wins,
                win_rate_pct=100.0 * wins / len(rows), mean_return=statistics.mean(returns),
                return_std=statistics.stdev(returns) if len(rows) > 1 else 0.0,
                elimination_wins=sum(bool(row["is_win"]) and int(row["end_reason"]) == 1 for row in rows),
                timeout_wins=sum(bool(row["is_win"]) and int(row["end_reason"]) == 2 for row in rows),
                pure_timeouts=sum(int(row["end_reason"]) == 2 for row in rows))


def refresh_eval_summary(run_dir):
    """Rebuild from immutable JSON under one lock; retries cannot duplicate rows."""
    from tabx_storage import read_json, resolve_under
    run_dir = Path(run_dir).resolve()
    with summary_lock(run_dir):
        rows = []
        for path in sorted((run_dir / "evaluations").glob("*.json")):
            report = read_json(path)
            snapshot = resolve_under(run_dir.parents[3], report["request"]["snapshot"])
            manifest = read_json(snapshot / "manifest.json")
            if manifest["run_id"] != report["request"]["run_id"] or manifest["agent_sha256"] != report["request"]["checkpoint_hash"]:
                raise ValueError("Evaluation summary checkpoint identity mismatch")
            rows.append(evaluation_summary(path.stem, report, manifest["t_env"]))
        rows.sort(key=lambda row: (row["checkpoint_step"], row["phase"], row["request_id"]))
        write_csv(run_dir / "eval_summary.csv", EVAL_FIELDS, rows)
        return rows


def update_eval_summary(run_dir, request_id, report, checkpoint_step):
    """Update one result; never reopen other checkpoints or evaluation JSON files."""
    run_dir = Path(run_dir)
    row = evaluation_summary(request_id, report, checkpoint_step)
    with summary_lock(run_dir):
        rows = {item["request_id"]: item for item in read_csv(run_dir / "eval_summary.csv")}
        rows[request_id] = row
        ordered = sorted(rows.values(), key=lambda item: (int(item["checkpoint_step"]), item["phase"], item["request_id"]))
        write_csv(run_dir / "eval_summary.csv", EVAL_FIELDS, ordered)
    return row


def reconcile_eval_summary(run_dir, rows_by_request_id):
    """Check one run after all reports validate; write at most once if different."""
    run_dir = Path(run_dir)
    path = run_dir / "eval_summary.csv"
    rows = sorted(rows_by_request_id.values(),
                  key=lambda item: (int(item["checkpoint_step"]), item["phase"], item["request_id"]))
    if not rows and not run_dir.exists():
        return "unchanged"
    # Serialize exactly as write_csv does: CSV strings must compare to serialized
    # numbers, rather than to the original int/float objects in the report.
    expected = io.StringIO(newline="")
    writer = csv.DictWriter(expected, fieldnames=EVAL_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    expected.seek(0)
    expected_rows = list(csv.reader(expected))
    with summary_lock(run_dir):
        try:
            with path.open("r", newline="", encoding="utf-8") as stream:
                current_rows = list(csv.reader(stream, strict=True))
        except FileNotFoundError:
            if not rows:
                return "unchanged"
            current_rows = None
        except (csv.Error, UnicodeError):
            current_rows = None
        if current_rows == expected_rows:
            return "unchanged"
        write_csv(path, EVAL_FIELDS, rows)
        return "written"


def summarize_training(run_dir):
    """Training-only live view; asynchronous results are merged from notifications."""
    from envs.tabx_outcome import END_REASONS
    from tabx_storage import read_json
    run_dir = Path(run_dir)
    metrics = read_csv(run_dir / "train_metrics.csv")
    episodes = read_episodes(run_dir / "training_episodes.jsonl")
    path = run_dir / "training_summary.json"
    training = read_json(path) if path.exists() else {}
    return dict(metrics=metrics, t_env=training.get("t_env", episodes[-1]["t_env"] if episodes else 0),
                learner_updates=training.get("learner_updates", int(metrics[-1]["learner_updates"]) if metrics else 0),
                elapsed_seconds=training.get("elapsed_seconds"), peak_cuda_bytes=training.get("peak_cuda_bytes"),
                last_end_reason=END_REASONS.get(int(episodes[-1]["end_reason"])) if episodes else None)


def summarize_run(run_dir):
    """Read-only live view, including the actual checkpoint step of each result."""
    from envs.tabx_outcome import END_REASONS
    from tabx_storage import read_json
    run_dir = Path(run_dir)
    metrics = read_csv(run_dir / "train_metrics.csv")
    evaluations = read_csv(run_dir / "eval_summary.csv")
    episodes = read_episodes(run_dir / "training_episodes.jsonl")
    summary_path = run_dir / "training_summary.json"
    training = read_json(summary_path) if summary_path.exists() else {}
    latest = max(evaluations, key=lambda row: (int(row["checkpoint_step"]), row["phase"], row["request_id"]), default={})
    tiers = [read_json(path) for path in sorted(run_dir.glob("selection/*/tiers.json"))]
    return dict(metrics=metrics, evaluations=evaluations, latest_evaluation=latest,
                t_env=training.get("t_env", episodes[-1]["t_env"] if episodes else 0),
                learner_updates=training.get("learner_updates", int(metrics[-1]["learner_updates"]) if metrics else 0),
                elapsed_seconds=training.get("elapsed_seconds"), peak_cuda_bytes=training.get("peak_cuda_bytes"),
                last_end_reason=END_REASONS.get(int(episodes[-1]["end_reason"])) if episodes else None,
                tiers=[dict(selection=str(path.parent.relative_to(run_dir)),
                            medium_q=tier["tiers"]["medium"]["q"], high_q=tier["tiers"]["high"]["q"])
                       for path, tier in zip(sorted(run_dir.glob("selection/*/tiers.json")), tiers)])


def train_line(rule, task, seed, gpu, row):
    losses = f"loss={float(row['loss']):.6g} "
    if row.get("total_loss") not in (None, ""):
        bc_loss = "" if row.get("bc_loss") in (None, "") else f"{float(row['bc_loss']):.6g}"
        losses += f"bc_loss={bc_loss} bc_weight={float(row['bc_weight']):.6g} total_loss={float(row['total_loss']):.6g} "
    # Old rows keep their original loss label and meaning, including mixed BC.
    return (f"[TRAIN] rule={rule} task={task} seed={seed} gpu={gpu} step={row['t_env']} "
            f"updates={row['learner_updates']} {losses}"
            f"td_abs={float(row['td_error_abs']):.6g} grad_norm_preclip={float(row['grad_norm_before_clip']):.6g} "
            f"grad_clip={float(row['grad_norm_clip']):g}")


def eval_line(rule, task, seed, row):
    return (f"[EVAL] rule={rule} task={task} seed={seed} checkpoint_step={row['checkpoint_step']} "
            f"phase={row['phase']} episodes={row['episodes']} wins={row['wins']} "
            f"win_rate={float(row['win_rate_pct']):.1f}% mean_return={float(row['mean_return']):.6g} "
            f"return_std={float(row['return_std']):.6g} elimination_wins={row['elimination_wins']} "
            f"timeout_wins={row['timeout_wins']}")
