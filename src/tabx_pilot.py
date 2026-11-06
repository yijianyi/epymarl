"""Lock family-disjoint splits, then stratify a pilot using training history only."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import statistics

from envs.tabx_tasks import file_sha256, index_bank, task_hash
from tabx_storage import atomic_json, read_json, stable_seed

SCHEMA_VERSION = "tabx-pilot-2"
RESERVED_TASKS = ("23", "651", "657", "647")
SPLITS = ("train", "validation", "test")
TIMEOUT_GROUPS = ("zero", "low", "medium", "high")
SCOPE_RANK = {"file": 1, "namespace": 2, "bank_hash": 3}
LINEAGE_KEYS = {"parent", "parents", "parent_id", "parent_ids", "parent_task_id", "parent_task_ids",
                "parent_hash", "parent_hashes", "parent_task_hashes", "ancestor_ids", "ancestors"}


def _is_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _source_scope(value, inherited=None):
    """A generator kind/worker/candidate ID is NOT a source namespace."""
    if not isinstance(value, dict):
        return inherited
    for fields in (("source_bank_hash", "bank_hash"), ("source_namespace", "namespace", "dataset_id", "source_id"),
                   ("source_file", "file")):
        for field in fields:
            if value.get(field) is not None:
                kind = "bank_hash" if "hash" in field else ("file" if "file" in field else "namespace")
                if inherited and SCOPE_RANK[json.loads(inherited)[0]] > SCOPE_RANK[kind]:
                    return inherited
                return json.dumps([kind, str(value[field])], separators=(",", ":"))
    return inherited


def _task_scope(metadata):
    return _source_scope(metadata.get("merged_from"),
                         _source_scope(metadata.get("source"), _source_scope(metadata)))


def _parent_references(metadata):
    def leaves(value, scope, path):
        if isinstance(value, (str, int)):
            yield str(value), scope, path
        elif isinstance(value, list):
            for item in value:
                yield from leaves(item, scope, path)
        elif isinstance(value, dict):
            scope = _source_scope(value, _source_scope(value.get("source"))) or scope
            # Prefer a structured hash reference over a possibly ambiguous name.
            hashes = [value[k] for k in ("canonical_hash", "task_hash", "hash") if _is_hash(value.get(k))]
            if hashes:
                yield hashes[0].lower(), scope, path
            else:
                for key in ("original_task_id", "task_id", "id", "name"):
                    if key in value:
                        yield from leaves(value[key], scope, path)
                        break

    def walk(value, scope, path):
        if isinstance(value, dict):
            scope = _source_scope(value, scope)
            for key, item in value.items():
                location = f"{path}/{key}"
                if key in LINEAGE_KEYS:
                    yield from leaves(item, scope, location)
                elif isinstance(item, (dict, list)):
                    yield from walk(item, scope, location)
        elif isinstance(value, list):
            for item in value:
                yield from walk(item, scope, path + "[]")
    yield from walk(metadata, _task_scope(metadata), "metadata")


def build_families(tasks, hashes=None):
    hashes = hashes or {key: task_hash(task) for key, task in tasks.items()}
    parents = {key: key for key in tasks}
    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    def union(first, second):
        parents[find(first)] = find(second)

    aliases = defaultdict(set)
    for key, task in tasks.items():
        metadata = task.get("metadata", {})
        for digest in (hashes[key], metadata.get("canonical_hash")):
            if _is_hash(digest):
                aliases[("hash", digest.lower())].add(key)
        scope = _task_scope(metadata)
        if scope:
            original = metadata.get("merged_from", {}).get("original_task_id", metadata.get("original_task_id", key))
            aliases[("legacy", scope, str(original))].add(key)
    for values in aliases.values():
        ordered = sorted(values)
        for key in ordered[1:]:
            union(ordered[0], key)

    children = defaultdict(set)
    evidence = defaultdict(set)
    untrusted, no_lineage = [], []
    for key, task in tasks.items():
        refs = list(_parent_references(task.get("metadata", {})))
        if not refs:
            no_lineage.append(key)
        for value, scope, location in refs:
            if _is_hash(value):
                identity = ("hash", value.lower())
            elif scope:
                identity = ("legacy", scope, value)
            else:
                untrusted.append(dict(task_id=key, reference=value, location=location,
                                      reason="legacy_identifier_without_source"))
                continue
            children[identity].add(key)
            evidence[identity].add(location)

    parent_families, missing = [], []
    for identity, child_ids in sorted(children.items()):
        parent_ids = aliases.get(identity, set())
        related = sorted(child_ids | parent_ids)
        for key in related[1:]:
            union(related[0], key)
        record = dict(kind=identity[0], reference=identity[-1], scope=identity[1] if identity[0] == "legacy" else None,
                      children=sorted(child_ids), parents=sorted(parent_ids), task_ids=related,
                      evidence=sorted(evidence[identity]))
        parent_families.append(record)
        if not parent_ids:
            missing.append(record)
    groups = defaultdict(list)
    for key in sorted(tasks):
        groups[find(key)].append(key)
    return dict(groups=sorted(groups.values()), parent_families=parent_families,
                unresolved_lineage=missing, untrusted_lineage=untrusted, tasks_without_lineage_metadata=no_lineage)


def _external_assignment(split_file, tasks, families, output):
    supplied = read_json(split_file)
    splits = supplied.get("splits", supplied)
    assigned = {}
    for name in SPLITS:
        for value in splits[name]:
            key = str(value)
            if key not in tasks or key in assigned:
                raise ValueError(f"Unknown/duplicate split task {key}")
            assigned[key] = name
    if set(assigned) != set(tasks):
        raise ValueError("External split must cover every task exactly once")
    conflicts = [dict(task_ids=keys, assignments={key: assigned[key] for key in keys})
                 for keys in families["groups"] if len({assigned[k] for k in keys}) > 1]
    if conflicts:
        report_path = output.with_name(output.stem + ".conflicts.json")
        protected = {Path(split_file).resolve(), output.resolve()}
        suffix = 0
        while report_path.resolve() in protected:
            suffix += 1
            report_path = output.with_name(output.stem + f".conflicts-{suffix}.json")
        atomic_json(report_path, dict(status="external_split_family_conflict", external_split_sha256=file_sha256(split_file),
                                     conflicts=conflicts, original_split_preserved=True))
        ids = sorted({key for conflict in conflicts for key in conflict["task_ids"]})
        raise ValueError(f"External split separates relatives; task IDs: {', '.join(ids)}; report: {report_path}. Original split preserved.")
    return assigned


def historical_timeout(task):
    metadata = task.get("metadata", {})
    for path in (("confirmation", "original", "truncation_rate"), ("filter_eval", "truncation_rate"),
                 ("fast_filter_eval", "truncation_rate")):
        value = metadata
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            value = float(value)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid historical timeout rate for task {task['task_id']}")
            return value, ".".join(path)
    return 0.0, "missing_assumed_zero"


def timeout_strata(tasks, splits):
    history = {key: historical_timeout(task) for key, task in tasks.items()}
    positives = sorted(history[key][0] for key in splits["train"] if history[key][0] > 0)
    boundaries = statistics.quantiles(positives, n=3, method="inclusive") if len(positives) > 1 else positives * 2
    def group(rate):
        if rate == 0:
            return "zero"
        if not boundaries:
            return "high"  # Held-out values never set training thresholds.
        return "low" if rate <= boundaries[0] else ("medium" if rate <= boundaries[1] else "high")
    groups = {key: group(value[0]) for key, value in history.items()}
    counts = {name: {split: sum(groups[key] == name for key in splits[split]) for split in SPLITS}
              for name in TIMEOUT_GROUPS}
    return groups, dict(computed_from="train_only", positive_tertiles=boundaries, positive_training_tasks=len(positives),
                        quantile_method="inclusive_linear_interpolation", equal_values_stay_together=True,
                        intervals={"zero": "rate == 0", "low": "0 < rate <= q1", "medium": "q1 < rate <= q2", "high": "rate > q2"},
                        counts=counts, historical_rates={key: value[0] for key, value in history.items()},
                        history_sources=dict(Counter(value[1] for value in history.values())),
                        missing_history=[key for key, value in history.items() if value[1] == "missing_assumed_zero"],
                        use="historical scene selection only; remeasure actual timeout rates after QMIX training")


def build_pilot(bank, output, seed=0, count=32, split_file=None):
    output = Path(output)
    _, tasks, bank_hash = index_bank(bank)
    if not 1 <= count <= len(tasks):
        raise ValueError("pilot count must be between 1 and the number of tasks")
    hashes = {key: task_hash(task) for key, task in tasks.items()}
    families = build_families(tasks, hashes)
    reserved = [key for key in RESERVED_TASKS if key in tasks]
    if count < len(reserved):
        raise ValueError(f"Pilot must retain reserved tasks: {reserved}")
    if split_file:
        assigned = _external_assignment(split_file, tasks, families, output)
        outside_train = [key for key in reserved if assigned[key] != "train"]
        if outside_train:
            raise ValueError(f"External split places required pilot task IDs outside train: {outside_train}; original split preserved")
        split_origin = dict(kind="external", sha256=file_sha256(split_file))
    else:
        assigned = {}
        totals = Counter()
        targets = dict(train=0.8 * len(tasks), validation=0.1 * len(tasks), test=0.1 * len(tasks))
        ordered = sorted(families["groups"], key=lambda keys: (-len(keys), stable_seed(min(hashes[k] for k in keys), seed, "split", 0)))
        for keys in ordered:
            if set(keys).intersection(reserved):
                assigned.update({key: "train" for key in keys})
                totals["train"] += len(keys)
        for keys in ordered:
            if keys[0] in assigned:
                continue
            name = max(SPLITS, key=lambda name: targets[name] - totals[name])
            assigned.update({key: name for key in keys})
            totals[name] += len(keys)
        split_origin = dict(kind="family_balanced_80_10_10", reserved_training_tasks=reserved, targets=targets)
    splits = {name: sorted(key for key in tasks if assigned[key] == name) for name in SPLITS}
    for family in families["parent_families"]:
        if len({assigned[key] for key in family["task_ids"]}) != 1:
            raise AssertionError(f"Parent family crosses splits: {family['task_ids']}")
    if output.exists():
        existing = read_json(output)
        if (existing.get("schema_version") != SCHEMA_VERSION or existing["bank_hash"] != bank_hash or
                existing["seed"] != seed or len(existing["pilot"]) != count or existing["split_origin"] != split_origin):
            raise ValueError("Existing immutable pilot manifest has different inputs/version; use a new output path")
        return existing
    timeout_groups, stratification = timeout_strata(tasks, splits)
    def features(key):
        task, metadata = tasks[key], tasks[key].get("metadata", {})
        teams = [row[0] for row in task["scenario"]["teams"]]
        return {("layout", str(metadata.get("layout_archetype", "unknown"))),
                ("composition", str(metadata.get("composition_archetype", "unknown"))),
                ("roster", teams.count(0), teams.count(1)), ("zones", int(task["zone_scenario"]["n_zone"]))}
    remaining = set(splits["train"])
    if len(remaining) < count:
        raise ValueError("Not enough training tasks for pilot")
    chosen = list(reserved)
    remaining.difference_update(chosen)
    seen = Counter(feature for key in chosen for feature in features(key))
    bucket_seen = Counter(timeout_groups[key] for key in chosen)
    def pick(candidates):
        best = max(candidates, key=lambda key: (not any(f[0] == "layout" and seen[f] for f in features(key)),
                    sum(1 / (1 + seen[f]) for f in sorted(features(key))), stable_seed(hashes[key], seed, "pilot", 0), key))
        chosen.append(best)
        remaining.remove(best)
        seen.update(features(best))
        bucket_seen[timeout_groups[best]] += 1
    for name in TIMEOUT_GROUPS:
        candidates = [key for key in remaining if timeout_groups[key] == name]
        if len(chosen) < count and not bucket_seen[name] and candidates:
            pick(candidates)
    while len(chosen) < count:
        available = {timeout_groups[key] for key in remaining}
        minimum = min(bucket_seen[name] for name in available)
        pick([key for key in remaining if bucket_seen[timeout_groups[key]] == minimum])
    for name in TIMEOUT_GROUPS:
        stratification["counts"][name]["pilot"] = bucket_seen[name]
    manifest = dict(schema_version=SCHEMA_VERSION, bank_hash=bank_hash, seed=seed, split_origin=split_origin,
                    splits=splits, pilot=chosen, task_hashes=hashes, lineage_groups=families["groups"],
                    parent_families=families["parent_families"], unresolved_lineage=families["unresolved_lineage"],
                    untrusted_lineage=families["untrusted_lineage"], tasks_without_lineage_metadata=families["tasks_without_lineage_metadata"],
                    lineage_audit=dict(trusted_parent_identifiers=len(families["parent_families"]), crossing_parent_families=0),
                    timeout_groups=timeout_groups, timeout_stratification=stratification,
                    pilot_features={key: sorted(map(str, features(key))) for key in chosen},
                    target_episodes=count * 2 * 2 * 20, rules=["hp", "loss"])
    atomic_json(output, manifest)
    return manifest
