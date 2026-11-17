"""Distance partitions, preserved learning, sealed targets and fixed-state RNN replay."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from tabx_storage import TARGET_FILES, atomic_json, file_sha256, read_json, target_snapshot_info
from tabx_td_success import DISTANCE_BINS, distance_metrics, gradient_norm
from test_tabx_td_success import teacher_pool


def test_distance_partitions_returns_padding_and_weighted_loss():
    lengths = [1, 5, 20, 50, 63]
    q = torch.arange(64.).reshape(1, -1, 1).repeat(5, 1, 1).requires_grad_()
    targets, rewards = torch.ones_like(q)*2, torch.ones_like(q)*9999
    mask, terminated = torch.zeros_like(q), torch.zeros_like(q)
    for i, length in enumerate(lengths):
        rewards[i, :length] = 0
        rewards[i, length-1] = 1
        mask[i, :length] = 1
        terminated[i, length-1] = 1
    groups = distance_metrics(q, targets, rewards, mask, terminated, .9, (1, 2, 2), torch.tensor([1, 1, 0, 1, 0]))
    total, squared = 0, 0.
    for source in ("legacy_demo", "success_td", "online"):
        group = groups[source]
        assert sum(group[name]["valid_steps"] for name, _, _ in DISTANCE_BINS) == group["all"]["valid_steps"]
        weighted = sum(group[name]["td_loss"]*group[name]["valid_steps"] for name, _, _ in DISTANCE_BINS if group[name]["valid_steps"])
        assert weighted/group["all"]["valid_steps"] == pytest.approx(group["all"]["td_loss"])
        assert group["terminal"]["reward_mean"] == group["terminal"]["trajectory_return_mean"] == 1
        total += group["all"]["valid_steps"]
        squared += weighted
    assert total == sum(lengths)
    assert squared/total == pytest.approx(float((((q-targets)*mask)**2).sum()/mask.sum()))
    assert groups["online_win"]["all"]["valid_steps"] == 50
    assert groups["online_loss"]["all"]["valid_steps"] == 63
    assert groups["online"]["far"]["valid_steps"] == 13
    assert groups["online"]["far"]["trajectory_return_mean"] == pytest.approx(np.mean(.9**np.arange(50, 63)))
    assert q.grad is None
    unavailable = distance_metrics(q, None, rewards, mask, terminated, .9, (0, 5, 0), torch.ones(5))
    assert unavailable["success_td"]["all"]["q_taken_mean"] is not None
    assert unavailable["success_td"]["all"]["target_mean"] is None
    with pytest.raises(ValueError, match="terminal"):
        distance_metrics(q, targets, rewards, mask, torch.zeros_like(terminated), .9, (0, 5, 0), torch.ones(5))


def test_gradients_distance_logging_and_targets_leave_updates_unchanged(teacher_pool, tmp_path):
    from controllers.basic_controller import BasicMAC
    from learners.q_learner import QLearner
    from tabx_dataset import episodes_to_batch
    from tabx_pipeline import QuietLogger
    from test_tabx_imitation import config
    _, episodes = teacher_pool
    batch = episodes_to_batch(episodes)[0]
    args = NS(**config(tabx_td_diagnostics=True, tabx_td_diagnostic_interval=1,
        tabx_run_dir=str(tmp_path), target_update_interval_or_tau=2))
    live = QLearner(BasicMAC(batch.scheme, batch.groups, args), batch.scheme, QuietLogger(), args)
    reference = deepcopy(live)
    reference.args = NS(**dict(vars(args), tabx_td_diagnostics=False))
    numpy_rng, torch_rng = np.random.get_state(), torch.get_rng_state().clone()
    for update in range(1, 3):
        live.train(batch, update*4, update, td_counts=(0, 2, 1))
        reference.train(batch, update*4, update)
        for actual, expected in zip(live.params, reference.params):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(live.target_mac.parameters(), reference.target_mac.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(numpy_rng[1], np.random.get_state()[1])
    assert torch.equal(torch_rng, torch.get_rng_state())
    rows = [json.loads(line) for line in (tmp_path/"td_distance_metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["parameter_updates"] == 0 and rows[1]["learner_update"] == 2
    assert rows[0]["timing"] == "before_optimizer_step_before_gradient_clip"
    assert all(rows[0]["gradient_norms"][key] > 0 for key in ("agent", "mixer", "mixer_v"))
    assert rows[0]["groups"]["online_loss"]["all"]["valid_steps"] == 4
    saved = tmp_path / "saved"
    saved.mkdir()
    live.save_models(str(saved))
    state = read_json(saved/"target_state.json")
    assert state["last_target_sync_update"] == 2 and state["target_sync_count"] == 1
    manifest = dict(learner_updates=2, last_target_sync_update=2, target_sync_count=1,
        **{key:file_sha256(saved/name) for name,key in TARGET_FILES})
    assert target_snapshot_info(saved, manifest)["available"]
    assert not target_snapshot_info(saved, {})["available"]
    with (saved/"target_agent.th").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="target_agent"):
        target_snapshot_info(saved, manifest)
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([3., 4.])
    assert gradient_norm([parameter]) == 5


def test_episode_cluster_uncertainty_does_not_treat_states_as_independent():
    from tabx_state_panel import cluster_summary
    def row(seed, delta):
        return dict(episode_seed=seed, paired_returns={"0.99":dict(delta_mean=delta)},
            q_gap_teacher_minus_model=-1., model_rollouts=[dict(is_win=0)], teacher_rollouts=[dict(is_win=1)])
    rows = [row(1, 0)]*20 + [row(2, 10)]
    summary = cluster_summary(rows, [.99], 7)
    result = summary["metrics"]["return_delta_gamma_0.99"]
    assert summary["episodes"] == 2 and summary["states"] == 21
    assert result["episode_weighted_mean"] == 5 and result["ci95"] == [0., 10.]
    assert cluster_summary(rows[:20], [.99], 7)["metrics"]["win_delta"]["ci95"] is None


@pytest.mark.skipif(os.environ.get("TABX_PROPAGATION_SMOKE") != "1", reason="Opt in with TABX_PROPAGATION_SMOKE=1")
def test_real_diagnostics_old_and_new_targets_and_own_rnn_histories(tmp_path):
    from tabx_pipeline import parser
    from tabx_propagation_diagnostic import diagnose_propagation, load_diagnostic_models
    from tabx_state_panel import create_value_panel, evaluate_value_panel, replay_state
    from tabx_expert import HeuristicTeacher
    from test_tabx_official import tiny_task
    task = tiny_task()
    task["scenario"].update(healths=[[100.], [1.]], attack_damages=[[100.], [0.]],
        sight_angles=[[3.14], [3.14]], attack_cooldowns=[[.1], [.1]])
    bank, demos, output = tmp_path/"bank.json", tmp_path/"demos", tmp_path/"train"
    atomic_json(bank, dict(schema_version="1.0", tasks=[task]))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    def execute(command, name):
        log = tmp_path/f"{name}.log"
        with log.open("w") as stream:
            result = subprocess.run([sys.executable,*command], env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        assert result.returncode == 0, log.read_text()[-6000:]
    execute(["src/tabx_pipeline.py","collect-expert","--bank",str(bank),"--root",str(demos),"--task","tiny",
        "--episodes","4","--episode-limit","4","--epsilon","0"], "collect")
    execute(["src/main.py","--config=qmix_tabx","--env-config=tabx","with","env_args.key=tiny",f"env_args.task_bank={bank}",
        f"tabx_output_root={output}","tabx_run_id=source","seed=7","use_cuda=False","tabx_bc=True",f"tabx_demo_root={demos}",
        "tabx_bc_steps=4","tabx_bc_batch_size=2","tabx_bc_eval_episodes=2","tabx_dagger_episodes=1",
        "env_args.max_episode_steps=4","batch_size=2","buffer_size=16","tabx_demo_decay_steps=16","tabx_demo_fraction=0.5",
        "tabx_td_success_fraction=0.25","tabx_td_diagnostics=True","tabx_td_diagnostic_interval=1",
        "target_update_interval_or_tau=2","save_model_interval=8","t_max=24"], "train")
    run = output/"hp/tiny/7/source"
    checkpoints = sorted((run/"checkpoints").iterdir(), key=lambda p:int(p.name))
    initial, final = checkpoints[0], checkpoints[-1]
    # A genuine old checkpoint lacks target data; never substitute its live network for historical targets.
    old = tmp_path/"old"
    shutil.copytree(final, old)
    manifest = read_json(old/"manifest.json")
    for name, key in TARGET_FILES:
        (old/name).unlink()
        manifest.pop(key)
    atomic_json(old/"manifest.json", manifest)
    source_hashes = {name:file_sha256(final/name) for name,_ in TARGET_FILES}
    inventory = tmp_path/"fixed-trajectories.json"
    arguments = ["diagnose-propagation","--bank",str(bank),"--snapshots",str(initial),str(final),str(old),
        "--demo-dir",str(demos/"hp/tiny/advanced"),"--panel",str(inventory),"--output",str(tmp_path/"propagation")]
    diagnose_propagation(parser().parse_args(arguments))
    report = read_json(tmp_path/"propagation/report.json")
    panel = read_json(inventory)
    assert panel["episodes"] == 4
    assert all(row["shard_sha256"] and row["trajectory_sha256"] for row in panel["trajectories"])
    assert [row["target_status"]["available"] for row in report["checkpoints"]] == [True,True,False]
    old_stats = report["checkpoints"][-1]["groups"]["all"]
    assert old_stats["q_taken_mean"] is not None and old_stats["target_mean"] is None
    assert "cannot be reconstructed" in report["checkpoints"][-1]["target_status"]["limitation"]
    assert report["checkpoints"][1]["groups"]["all"]["q_taken_mean"] == old_stats["q_taken_mean"]
    for name,digest in source_hashes.items():
        assert file_sha256(final/name) == digest
    # Reusing the same inventory with another chunk size retains exactly the same episode identities.
    arguments[-1] = str(tmp_path/"propagation2")
    diagnose_propagation(parser().parse_args(arguments+["--batch-size","1"]))
    assert read_json(inventory) == panel
    state_panel = tmp_path/"states.json"
    create_value_panel(parser().parse_args(["value-panel","--bank",str(bank),"--snapshots",str(initial),str(final),
        "--panel",str(state_panel),"--states","2","--episodes","2","--spacing","1","--include-search-turns"]))
    states = read_json(state_panel)["states"]
    assert len(states) == 4 and all("hidden" not in row and "hidden_sha256" not in row for row in states)
    evaluate_value_panel(parser().parse_args(["diagnose-value-panel","--bank",str(bank),"--panel",str(state_panel),
        "--snapshots",str(initial),str(final),"--output",str(tmp_path/"value"),"--repeats","2"]))
    value = read_json(tmp_path/"value/report.json")
    assert len(value["checkpoints"]) == 2
    for checkpoint in value["checkpoints"]:
        assert checkpoint["cluster_summary"]["episodes"] >= 1
        for row in checkpoint["states"]:
            assert row["action_diagnostics"]["valid_alive_agents"] == 1
            assert row["state_sha256"] and "delta_standard_error" in row["paired_returns"]["0.99"]
            for a,b in zip(row["model_rollouts"],row["teacher_rollouts"]):
                assert a["seed"] == b["seed"]
                if row["model_actions"] == row["teacher_actions"]:
                    assert a["rewards"] == b["rewards"] and a["is_win"] == b["is_win"]
    # Recompute one prefix twice with the same model, then change its weights: environment checksum
    # stays fixed while the newly computed hidden vector changes, proving it is not reused from the panel.
    runner,mixer,_,_,_ = load_diagnostic_models(final,bank,"hp")
    teacher = HeuristicTeacher(runner.env,"advanced",0.)
    with torch.inference_mode():
        first = replay_state(runner,mixer,teacher,states[0])
        second = replay_state(runner,mixer,teacher,states[0])
        assert first["hidden_sha256"] == second["hidden_sha256"]
        for parameter in runner.mac.parameters():
            parameter.add_(.1)
        third = replay_state(runner,mixer,teacher,states[0])
        assert first["state_sha256"] == third["state_sha256"]
        assert first["hidden_sha256"] != third["hidden_sha256"]
        altered = dict(states[0],state_sha256="bad")
        with pytest.raises(ValueError,match="checksum"):
            replay_state(runner,mixer,teacher,altered)
    runner.close_env()
