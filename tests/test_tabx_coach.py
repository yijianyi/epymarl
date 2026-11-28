"""Regression tests for ordered B/C/D admission and paired coach selection."""
import os
from types import SimpleNamespace

import pytest

from tabx_coach_collection import load_coach_tiers
from tabx_coach_scheduler import CoachScheduler, coach_jobs
from tabx_coach_selection import (MEDIUM_PARAMETERS, SCHEMA, confirm_selection, eligible_snapshot, pure_ppo_start,
                                 screen_candidates, selection_path)
from tabx_medium_cli import parser
from tabx_storage import SEED_PROTOCOL, atomic_json


def test_coach_defaults_and_common_window():
    args = parser("coach").parse_args(["--bank", "bank.json", "--tasks", "25,147", "--gpus", "0"])
    assert (args.seeds, args.t_max, args.batch_size, args.bc_steps, args.jobs_per_gpu) == ("1", 3000000, 32, 10000, 12)
    assert (args.screen_episodes, args.confirm_episodes, args.save_model_interval) == (32, 128, 25000)
    mix = dict(tabx_mix_start=args.mix_start, tabx_mix_decay_steps=args.mix_decay_steps,
               tabx_mappo_temp_peak=1.0, tabx_mappo_temp_end_step=args.temp_end_step)
    assert pure_ppo_start([mix]) == 50_000
    assert pure_ppo_start([mix, dict(mix, tabx_mappo_temp_peak=args.temp_peak)]) == 100_000
    assert pure_ppo_start([dict(mix, tabx_mix_start=0.0)]) == 0
    manifest = dict(policy_stage="online_ppo", batch_start_t_env=98_000, t_env=105_000,
                    learner_updates=42, teacher_probability=0.0, policy_temperature=1.0)
    assert not eligible_snapshot(manifest, 100_000)
    pure = dict(manifest, batch_start_t_env=100_000)
    assert eligible_snapshot(pure, 100_000)
    assert not eligible_snapshot(dict(pure, policy_stage="after_bc_before_online", t_env=0), 100_000)
    assert not eligible_snapshot(dict(pure, learner_updates=0), 100_000)
    assert not eligible_snapshot(dict(pure, teacher_probability=0.001), 100_000)
    assert not eligible_snapshot(dict(pure, policy_temperature=1.001), 100_000)


def stats(mean, wins, episodes=8):
    return dict(mean_return=mean, wins=wins, episodes=episodes)


def test_screen_deduplicates_weights_and_uses_shared_negative_return_scale():
    values = [("B", "actor", 3, -3, 3), ("B", "actor", 2, -3, 3), ("B", "actor", 1, -3, 2),
              ("B", "far", 5, -5, 3), ("C", "half", 4, -4, 4), ("C", "best", 3, -2, 3),
              ("D", "solo", 2, -2.1, 2)]
    screens = [dict(group=group, checkpoint_hash=actor, t_env=step, train_seed=1, run_id=group,
                    screen=stats(reward, wins)) for group, actor, step, reward, wins in values]
    groups = {group: {} for group in "BCD"}
    candidates, reference = screen_candidates(screens, groups, stats(-6, 1))
    assert reference["top_mean_return"] == -2
    assert [(c["group"], c["role"], c["t_env"], c["q_screen"]) for c in candidates] == [
        ("B", "best", 2, 0.75), ("B", "half", 5, 0.25), ("C", "best", 3, 1.0),
        ("C", "half", 4, 0.5), ("D", "best", 2, pytest.approx(0.975))]
    assert groups["B"]["missing_half_reason"] == "outside_screen_half_range"
    assert groups["D"]["screened_distinct_weights"] == 1
    assert groups["D"]["missing_half_reason"] == "fewer_than_two_distinct_screened_weights"
    candidates, _ = screen_candidates(screens, groups, stats(-2, 2))
    assert len(candidates) == 3
    assert groups["B"]["missing_half_reason"] == "missing_positive_screen_gap"


def test_confirmation_uses_paired_wins_and_can_select_different_groups():
    values = [("B", "best", "actor-b", -4, 5), ("B", "half", "low-b", -5, 3),
              ("C", "best", "actor-c", -2, 4), ("C", "half", "low-c", -5, 5),
              ("D", "best", "actor-d", -3, 6), ("D", "half", "low-d", -4, 5)]
    candidates = [dict(group=group, role=role, checkpoint_hash=actor, t_env=20, train_seed=1, run_id=group,
                       confirm=stats(reward, wins)) for group, role, actor, reward, wins in values]
    selected, reasons, _ = confirm_selection(candidates, stats(-4, 4), stats(-6, 1))
    assert reasons == dict(half=None, high=None)
    assert selected["high"]["group"] == "D"
    assert selected["half"]["group"] == "B"
    assert selected["half"]["q_confirm"] == 0.5
    assert candidates[1]["medium_comparison"] == "below_medium"
    assert candidates[1]["win_rate_diff_pp"] == -12.5
    assert candidates[2]["medium_comparison"] == "tied_medium"
    assert sum(c["collectable"] for c in candidates) == 2

    selected, reasons, _ = confirm_selection(candidates, stats(-4, 5), stats(-6, 1))
    assert selected["high"]["group"] == "D" and selected["half"] is None and reasons["half"]
    selected, reasons, _ = confirm_selection(candidates, stats(-4, 6), stats(-6, 1))
    assert all(item is None for item in selected.values())
    assert reasons["high"] == "no_candidate_above_medium"


def test_collection_skips_an_older_selection_with_incomplete_training(tmp_path):
    official = dict(layout_version="layout", simulator_contract="contract",
                    parameter_json_sha256={MEDIUM_PARAMETERS: "medium"})
    directory = tmp_path / "_batches/hp/coach"
    batch = dict(batch_id="coach", rule="hp", bank_hash="bank", task_hashes={"25": "task"},
                 training_config=dict(env_args=dict(max_episode_steps=8)))
    path = selection_path(directory, "25")
    atomic_json(path, dict(schema_version=SCHEMA, batch_id="coach", task_id="25", task_hash="task",
                           bank_hash="bank", rule="hp", episode_limit=8, medium_parameter_sha256="medium",
                           official_tabx=official, layout_version="layout", simulator_contract="contract",
                           seed_protocol=SEED_PROTOCOL, action_selection="greedy", evaluation_epsilon=0.0,
                           selected=dict(high=dict(collectable=True, medium_comparison="above_medium"), half=None),
                           missing_reasons=dict(high=None, half="missing_half"),
                           groups={"B": dict(incomplete_runs=[]), "C": dict(incomplete_runs=[]),
                                   "D": dict(incomplete_runs=[dict(train_seed=1, status="failed")])}))
    selected_path, _, jobs, reasons = load_coach_tiers(tmp_path, directory, batch, "25", None, official)
    assert selected_path == path
    assert jobs == []
    assert reasons == dict(high="training_incomplete", half="training_incomplete")


@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    args = SimpleNamespace(root=str(tmp_path), rule="hp", gpus=["0", "1"], jobs_per_gpu=2, batch_id="coach")
    jobs = [dict(spec, status="pending", needs_training=True, attempts=[]) for spec in coach_jobs(["25", "147"], [1])]
    batch = CoachScheduler(args, tmp_path / "_batches/hp/coach", dict(training_config={}), dict(jobs=jobs))
    batch.bundles.extend({job["group"]: job for job in jobs if job["task_id"] == task} for task in ("25", "147"))
    batch.initialize_job_cache()
    batch.pool.register()
    batch.started = []
    batch.sealed = set()
    children = {}

    def launch(job, gpu):
        task, group = job["task_id"], job["group"]
        key = f"{task}-{group}"
        batch.started.append(key)
        job["attempts"].append(dict(run_id=key, gpu=gpu, status="running"))
        job["needs_training"] = False
        batch.running[key] = None
        children[key] = os.dup(batch.lease[-1])

    def finish(key):
        del batch.running[key]
        os.close(children.pop(key))

    monkeypatch.setattr(batch, "launch", launch)
    monkeypatch.setattr(batch, "demo_ready", lambda task: True)
    monkeypatch.setattr(batch, "source_ready", lambda job: job["task_id"] in batch.sealed)
    batch.finish = finish
    try:
        yield batch
    finally:
        batch.release_reservations()
        batch.pool.unregister()
        for fd in children.values():
            os.close(fd)


def test_admits_task_trios_in_input_order_without_waiting_for_training_to_finish(scheduler):
    scheduler.fill_slots()
    assert scheduler.started == ["25-B"]
    assert list(scheduler.reserved) == ["C", "D"]
    scheduler.sealed.add("25")
    scheduler.fill_slots()
    assert scheduler.started == ["25-B", "25-C", "25-D"]
    assert not scheduler.reserved
    scheduler.finish("25-B")
    scheduler.finish("25-C")
    scheduler.fill_slots()
    assert scheduler.started[-1] == "147-B" and "25-D" in scheduler.running
    scheduler.sealed.add("147")
    scheduler.fill_slots()
    assert scheduler.started[-2:] == ["147-C", "147-D"]


@pytest.mark.parametrize("sealed", [False, True])
def test_b_failure_releases_reservations_or_allows_sealed_dependents(scheduler, sealed):
    scheduler.fill_slots()
    scheduler.finish("25-B")
    if sealed:
        scheduler.sealed.add("25")
    scheduler.fill_slots()
    if sealed:
        assert not any(job.get("dependency_error") for job in scheduler.jobs[1:3])
        assert scheduler.started == ["25-B", "25-C", "25-D"]
    else:
        assert all("B failed before sealing" in job["dependency_error"] for job in scheduler.jobs[1:3])
        assert scheduler.started == ["25-B", "147-B"]
