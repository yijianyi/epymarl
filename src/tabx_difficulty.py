"""Shared, evidence-based TABX opponent identification; unknown is never medium."""
import hashlib
import json
from pathlib import Path

DIFFICULTIES = ("medium", "advanced")
# Audited immutable source trees: build_task_params loads a literal preset and
# TABXWrapper passes it directly to the simulator. A parameter inventory alone
# cannot establish which preset was actually used.
LEGACY_SIMULATOR_SOURCE_HASH = "d42f6b18f28932bac661341f1891e6a63602b2de290e3dcc649fdfdec6c841fe"
LEGACY_PRESETS = {
    # 31e85b1: fixed medium opponent
    "fa9a5845143cc85a5edf80def43e6a9ee7266e59dec0788cac1d9eb23dbc3b30":
        ("medium", "1db27971cf931d9bb3beb7b9869ff2df6d8d8d26b26308fecb517c6e4f3ef0f6"),
    # 9dae7c6: fixed advanced opponent
    "3f008c1bfebea17352f638458031260a9f7bb49fabf060d0f7aa7c45e2c65005":
        ("advanced", "c639cf4df1062146da7ec74d314a9645b2bc022d5f0905a11bfae156dc8b2edb"),
}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _explicit(records, context):
    evidence = {}
    for name, value in records:
        if value in (None, "unknown"):
            continue
        if value not in DIFFICULTIES:
            raise ValueError(f"{context}: invalid opponent difficulty {name}={value!r}")
        evidence[name] = value
    return evidence


def _resolve(evidence, context):
    if len(set(evidence.values())) > 1:
        raise ValueError(f"{context}: opponent difficulty mismatch/conflict: {evidence}")
    return dict(difficulty=next(iter(evidence.values()), "unknown"), evidence=evidence)


def _legacy(record, label):
    provenance = record.get("provenance") or {}
    source = (provenance.get("epymarl") or {}).get("source_hash")
    known = LEGACY_PRESETS.get(source)
    parameters = (provenance.get("tabx") or {}).get("parameter_json_sha256") or {}
    presets = {path: digest for path, digest in parameters.items()
               if path.startswith("src/tabx/heuristic_policy/parameters/") and path.endswith(".json")}
    if (known and (provenance.get("tabx") or {}).get("source_hash") == LEGACY_SIMULATOR_SOURCE_HASH
            and presets == {f"src/tabx/heuristic_policy/parameters/{known[0]}.json": known[1]}):
        return {f"{label}.audited_source:{source}": known[0]}
    return {}


def checkpoint_difficulty(snapshot, *, manifest=None, config=None, context=None):
    snapshot = Path(snapshot)
    context = context or f"checkpoint={snapshot}"
    if manifest is None:
        manifest = _read(snapshot / "manifest.json") if (snapshot / "manifest.json").is_file() else {}
    if config is None:
        config = _read(snapshot / "config.json") if (snapshot / "config.json").is_file() else {}
    if manifest.get("config_sha256") and (snapshot / "config.json").is_file():
        digest = hashlib.sha256((snapshot / "config.json").read_bytes()).hexdigest()
        if digest != manifest["config_sha256"]:
            raise ValueError(f"{context}: checkpoint config SHA-256 mismatch")
    evidence = _explicit((("manifest", manifest.get("opponent_difficulty")),
                          ("config.env_args", (config.get("env_args") or {}).get("opponent_difficulty"))), context)
    evidence.update(_legacy(manifest, "manifest.provenance"))
    return _resolve(evidence, context)


def evaluation_difficulty(report, *, snapshot=None, checkpoint_step=None, request_id=None):
    request = report["request"]
    context = f"request_id={request_id or 'unknown'} checkpoint={snapshot or request.get('snapshot', 'unknown')}"
    evidence = _explicit((("report", report.get("opponent_difficulty")),
                          ("request", request.get("opponent_difficulty"))), context)
    # New reports are authoritative about the actual evaluator. No historical
    # checkpoint reads are needed on the normal per-result hot path.
    if evidence:
        return _resolve(evidence, context)
    evidence.update(_legacy(report, "report.provenance"))
    if snapshot is None and request.get("snapshot") and Path(request["snapshot"]).is_absolute():
        snapshot = Path(request["snapshot"])
    if snapshot is not None:
        snapshot = Path(snapshot)
        manifest = _read(snapshot / "manifest.json") if (snapshot / "manifest.json").is_file() else {}
        fields = (("checkpoint_hash", "agent_sha256"), ("run_id", "run_id"), ("task_id", "task_id"),
                  ("task_hash", "task_hash"), ("bank_hash", "bank_hash"), ("rule", "timeout_rule"),
                  ("train_seed", "train_seed"))
        for field, key in fields:
            if field in request and key in manifest and request[field] != manifest[key]:
                raise ValueError(f"{context}: checkpoint identity mismatch: {field}")
        if checkpoint_step is not None and manifest.get("t_env", checkpoint_step) != int(checkpoint_step):
            raise ValueError(f"{context}: checkpoint step mismatch")
        bound = (request.get("run_id") is not None and request.get("checkpoint_hash") is not None
                 and manifest.get("run_id") == request["run_id"]
                 and manifest.get("agent_sha256") == request["checkpoint_hash"])
        config = None
        if not bound and request.get("run_id") and request.get("checkpoint_hash") and (snapshot / "agent.th").is_file():
            # If the manifest was lost, the actual weights and source run path
            # can still bind the adjacent config to this historical evaluation.
            config = _read(snapshot / "config.json") if (snapshot / "config.json").is_file() else {}
            run_id = config.get("tabx_run_id") or snapshot.parent.parent.name
            digest = hashlib.sha256((snapshot / "agent.th").read_bytes()).hexdigest()
            if run_id != request["run_id"] or digest != request["checkpoint_hash"]:
                raise ValueError(f"{context}: checkpoint identity mismatch: run_id/agent SHA-256")
            for field, value in (("task_id", (config.get("env_args") or {}).get("key")),
                                 ("rule", (config.get("env_args") or {}).get("timeout_rule")),
                                 ("train_seed", config.get("seed"))):
                if field in request and value is not None and request[field] != value:
                    raise ValueError(f"{context}: checkpoint config identity mismatch: {field}")
            bound = True
        if bound:
            evidence.update({f"checkpoint.{key}": value for key, value in
                             checkpoint_difficulty(snapshot, manifest=manifest, config=config, context=context)["evidence"].items()})
    return _resolve(evidence, context)


def select_checkpoint(path, load_step=0):
    """Select once; all later identification, validation and loading use this path."""
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")
    if (path / "agent.th").is_file():
        return path
    steps = sorted(int(child.name) for child in path.iterdir() if child.is_dir() and child.name.isdigit())
    if not steps:
        raise ValueError(f"No checkpoints found in {path}")
    step = steps[-1] if int(load_step) == 0 else min(steps, key=lambda step: (abs(step - int(load_step)), step))
    return path / str(step)


def requested_difficulty(snapshot, requested=None):
    """Ordinary restoration: inherit, verify an explicit choice, or demand one."""
    context = f"checkpoint={snapshot} requested={requested!r}"
    if requested is not None and requested not in DIFFICULTIES:
        raise ValueError(f"{context}: explicit difficulty must be medium or advanced")
    identified = checkpoint_difficulty(snapshot, context=context)
    source = identified["difficulty"]
    if requested is not None and source != "unknown" and requested != source:
        raise ValueError(f"{context}: opponent difficulty mismatch: source={source}, requested={requested}")
    if source == "unknown" and requested is None:
        raise ValueError(f"{context}: opponent difficulty is unknown; explicitly set env_args.opponent_difficulty")
    return dict(difficulty=source if requested is None else requested,
                source="checkpoint" if requested is None else "launch_argument",
                checkpoint_difficulty=source, evidence=identified["evidence"])
