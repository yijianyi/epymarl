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


class TrainingProcess:
    pid = 0

    def __init__(self, pass_fds):
        self.fd = os.dup(pass_fds[0])
        self.returncode = None

    def poll(self):
        return self.returncode

    def exit(self):
        if self.returncode is None:
            os.close(self.fd)
            self.returncode = 17


@pytest.fixture
def make_scheduler(tmp_path, monkeypatch):
    batches = []

    def make(tasks, *, gpus=("0", "1"), jobs_per_gpu=2, seeds=(1,), demos=None):
        args = SimpleNamespace(root=str(tmp_path), rule="hp", gpus=list(gpus), jobs_per_gpu=jobs_per_gpu,
                               batch_id="coach", resume=False, eval_workers=1, poll_seconds=0.01)
        jobs = [dict(spec, status="pending", attempts=[]) for spec in coach_jobs(tasks, seeds)]
        batch = CoachScheduler(args, tmp_path / "_batches/hp/coach", dict(training_config={}), dict(jobs=jobs))
        batch.pool.register()
        batches.append(batch)
        batch.prepare_resume()
        ready_demos = set(tasks if demos is None else demos)
        sealed, started, children = set(), [], {}
        launch = batch.launch

        def record_launch(job, gpu):
            launch(job, gpu)
            key = (job["task_id"], job["train_seed"], job["group"])
            started.append(key)
            children[key] = batch.running[job["attempts"][-1]["run_id"]][0]

        def finish(task, group="B", seed=1):
            children[task, seed, group].exit()
            assert batch.poll_training()

        monkeypatch.setattr("tabx_scheduler.subprocess.Popen",
                            lambda *args, pass_fds, **kwargs: TrainingProcess(pass_fds))
        monkeypatch.setattr(batch, "training_command", lambda job, attempt: [])
        monkeypatch.setattr(batch, "launch", record_launch)
        monkeypatch.setattr(batch, "demo_ready", lambda task: task in ready_demos)
        monkeypatch.setattr(batch, "source_ready", lambda job: (job["task_id"], job["train_seed"]) in sealed)
        return SimpleNamespace(batch=batch, started=started, sealed=sealed, finish=finish, ready_demos=ready_demos)

    yield make

    for batch in batches:
        for process, stream, _ in batch.running.values():
            process.exit()
            stream.close()
        batch.dispatcher.close()
        batch.log.close()
        batch.pool.unregister()


def test_four_slots_start_four_b_jobs_and_refill_one_at_a_time(make_scheduler):
    tasks = ["25", "147", *(str(task) for task in range(200, 230))]
    scheduler = make_scheduler(tasks)
    batch = scheduler.batch
    batch.fill_slots()
    assert scheduler.started == [(task, 1, "B") for task in tasks[:4]]
    assert batch.gpu_in_use == {"0": 2, "1": 2}
    ready = [batch.jobs_by_id[job_id] for job_id in batch.pending_train
             if batch.jobs_by_id[job_id]["dependency_status"] == "waiting_gpu"]
    assert [job["task_id"] for job in ready] == tasks[4:]
    assert all(job["group"] == "B" for job in ready)

    scheduler.sealed.add(("25", 1))
    batch.fill_slots()
    ready_queue = [batch.jobs_by_id[job_id] for job_id in list(batch.pending_train)[:30]]
    assert [(job["task_id"], job["group"]) for job in ready_queue] == (
        [(task, "B") for task in tasks[4:]] + [("25", "C"), ("25", "D")])
    scheduler.sealed.add(("147", 1))
    scheduler.finish("147")
    assert scheduler.started == [(task, 1, "B") for task in tasks[:5]]
    assert batch.gpu_in_use == {"0": 2, "1": 2}


def test_single_slot_runs_b_then_c_and_d(make_scheduler):
    scheduler = make_scheduler(["25"], gpus=("0",), jobs_per_gpu=1)
    batch = scheduler.batch
    batch.fill_slots()
    assert scheduler.started == [("25", 1, "B")]
    assert [job["dependency_status"] for job in batch.jobs[1:]] == ["waiting_step0", "waiting_step0"]
    scheduler.sealed.add(("25", 1))
    batch.fill_slots()
    assert [job["dependency_status"] for job in batch.jobs[1:]] == ["waiting_gpu", "waiting_gpu"]
    scheduler.finish("25")
    assert scheduler.started == [("25", 1, "B"), ("25", 1, "C")]
    scheduler.finish("25", "C")
    assert scheduler.started == [("25", 1, group) for group in "BCD"]


def test_waiting_demo_does_not_block_other_b_or_overtake_ready_jobs(make_scheduler):
    scheduler = make_scheduler(["25", "147"], gpus=("0",), jobs_per_gpu=1, demos=["147"])
    batch = scheduler.batch
    batch.fill_slots()
    assert scheduler.started == [("147", 1, "B")]
    assert batch.jobs[0]["dependency_status"] == "waiting_demo"
    scheduler.sealed.add(("147", 1))
    batch.fill_slots()
    scheduler.ready_demos.add("25")
    batch.fill_slots()

    scheduler.finish("147")
    assert scheduler.started[-1] == ("147", 1, "C")
    scheduler.finish("147", "C")
    assert scheduler.started[-1] == ("147", 1, "D")
    scheduler.finish("147", "D")
    assert scheduler.started[-1] == ("25", 1, "B")


def test_c_and_d_can_start_while_their_b_is_running(make_scheduler):
    scheduler = make_scheduler(["25"], seeds=(9, 1))
    batch = scheduler.batch
    batch.fill_slots()
    assert scheduler.started == [("25", 9, "B"), ("25", 1, "B")]
    scheduler.sealed.add(("25", 9))
    batch.fill_slots()
    assert scheduler.started[2:] == [("25", 9, "C"), ("25", 9, "D")]
    assert len(batch.running) == 4
    assert [job["dependency_status"] for job in batch.jobs if job["group"] != "B" and job["train_seed"] == 1] == [
        "waiting_step0", "waiting_step0"]


@pytest.mark.parametrize("sealed", [False, True])
def test_b_failure_only_blocks_its_dependents_if_step0_is_missing(make_scheduler, sealed):
    scheduler = make_scheduler(["25", "147"])
    batch = scheduler.batch
    batch.fill_slots()
    if sealed:
        scheduler.sealed.add(("25", 1))
    scheduler.finish("25")
    if sealed:
        assert scheduler.started[2:] == [("25", 1, "C"), ("25", 1, "D")]
        assert all(not job.get("dependency_error") for job in batch.jobs[1:3])
    else:
        assert scheduler.started == [("25", 1, "B"), ("147", 1, "B")]
        assert all("B failed before sealing step 0: Training exited with code 17" == job["dependency_error"]
                   for job in batch.jobs[1:3])
        assert all(job["dependency_status"] == "dependency_failed" and not job["needs_training"]
                   for job in batch.jobs[1:3])
    assert all(job["dependency_status"] == "waiting_step0" for job in batch.jobs[4:])
