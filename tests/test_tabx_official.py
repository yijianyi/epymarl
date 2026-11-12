"""Official native-size conversion, identity slots and migration compatibility."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from envs.tabx_layout import TABXLayout, LAYOUT_VERSION, SIMULATOR_CONTRACT
from envs.tabx_outcome import adjudicate, replace_terminal_reward
from envs.tabx_tasks import import_tabx, validate_task_structure, official_tabx_root, file_sha256
from tabx_storage import atomic_json, ensure_output_contract, read_json, simulator_identity


@pytest.mark.parametrize("sizes", [(2, 3, 0), (10, 10, 2), (4, 4, 10), (5, 5, 5), (1, 10, 1), (10, 1, 0)])
def test_entity_identity_cyclic_observer_mapping_and_state(sizes):
    layout = TABXLayout(*sizes)
    slots = list(range(layout.n_ally)) + list(range(10, 10 + layout.n_enemy))
    for observer in range(layout.n_units):
        others = list(range(observer + 1, layout.n_units)) + list(range(observer))
        blocks = [entity * 100 + np.arange(16, dtype=np.float32) for entity in others]
        zones = 9000 + np.arange(layout.n_zone * 6, dtype=np.float32)
        native = np.r_[np.arange(14, dtype=np.float32) + observer, np.concatenate(blocks), zones]
        fixed = layout.pack_local(native, observer)
        fixed_order = list(range(slots[observer] + 1, 20)) + list(range(slots[observer]))
        for position, slot in enumerate(fixed_order):
            expected = slots.index(slot) * 100 + np.arange(16) if slot in slots else np.zeros(16)
            np.testing.assert_array_equal(fixed[14 + position * 16:14 + (position + 1) * 16], expected)
        np.testing.assert_array_equal(fixed[318:318 + len(zones)], zones)
        np.testing.assert_array_equal(layout.unpack_local(fixed, observer), native)
    native_state = np.arange(layout.native_state_size, dtype=np.float32) + 1
    fixed_state = layout.pack_state(native_state)
    for entity, slot in enumerate(slots):
        np.testing.assert_array_equal(fixed_state[slot * 14:(slot + 1) * 14], native_state[entity * 14:(entity + 1) * 14])
    np.testing.assert_array_equal(layout.unpack_state(fixed_state), native_state)
    assert np.isfinite(fixed_state).all()
    actions = np.arange(10) % 8
    np.testing.assert_array_equal(layout.native_actions(actions), actions[:layout.n_ally])
    masks = layout.pack_avail_actions(np.ones((layout.n_ally, 8)), np.zeros(layout.n_ally, dtype=bool))
    assert masks[:, 7].all() and np.all(masks.sum(axis=1) == 1)
    masks = layout.pack_avail_actions(np.ones((layout.n_ally, 8)), np.ones(layout.n_ally, dtype=bool))
    assert np.all(masks[:layout.n_ally] == 1)
    assert np.all(masks[layout.n_ally:].sum(axis=1) == 1)


def tiny_task():
    return dict(task_id="tiny", scenario=dict(positions=[[-2., 0.], [2., 0.]], rotations=[[0.], [3.]],
        body_weights=[[1.], [1.]], body_radiuss=[[1.], [1.]], teams=[[0], [1]],
        pos_min=[[-10., -10.], [-10., -10.]], pos_max=[[10., 10.], [10., 10.]], unit_ids=[[1], [9]],
        healths=[[10.], [10.]], attack_damages=[[2.], [-1.]], attack_ranges=[[3.], [3.]],
        attack_cooldowns=[[2.], [2.]], sight_angles=[[1.57], [1.57]], is_alive=[[True], [True]],
        attack_types=[[0], [1]], is_disabled=[[False], [False]], speeds=[[1.], [1.]]),
        zone_scenario=dict(n_zone=0, position=[], axes=[], zone_type=[], effect_value=[]))


@pytest.mark.parametrize("field,value", [("teams", [[1], [0]]), ("positions", [[99., 0.], [2., 0.]]),
    ("healths", [[float("nan")], [1.]]), ("body_radiuss", [[0.], [1.]]), ("speeds", [[-1.], [1.]]),
    ("attack_types", [[2], [1]]), ("is_alive", [[2], [True]]), ("teams", [0, 1])])
def test_reject_invalid_native_task(field, value):
    task = tiny_task()
    task["scenario"][field] = value
    with pytest.raises(ValueError):
        validate_task_structure(task)


def test_structural_audit_returns_failed_task_ids(tmp_path):
    from tabx_pipeline import check_bank
    good, bad = tiny_task(), tiny_task()
    bad["task_id"], bad["zone_scenario"]["n_zone"] = "bad", 11
    path, output = tmp_path / "bank.json", tmp_path / "report.json"
    atomic_json(path, dict(schema_version="1.0", tasks=[good, bad]))
    with pytest.raises(SystemExit):
        check_bank(NS(bank=str(path), output=str(output), bank_sha256=None))
    assert read_json(output)["failed_task_ids"] == ["bad"]


def test_only_official_checkout_is_imported_even_with_legacy_environment(tmp_path):
    legacy = tmp_path / "legacy"
    (legacy / "src/tabx").mkdir(parents=True)
    (legacy / "src/__init__.py").write_text("raise RuntimeError('legacy checkout imported')")
    code = "import sys;sys.path.insert(0,'src');from envs.tabx_tasks import import_tabx,official_tabx_root;from pathlib import Path;m=import_tabx();assert Path(m.__file__).resolve().parent==official_tabx_root().resolve()/'src/tabx'"
    subprocess.run([sys.executable, "-c", code], check=True,
                   env=dict(os.environ, TABX_ROOT=str(legacy), PYTHONPATH=str(legacy), JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES=""))


def test_missing_or_wrong_submodule_is_explicit(monkeypatch, tmp_path):
    import envs.tabx_tasks as tasks
    with pytest.raises(ValueError, match="official submodule"):
        import_tabx(tmp_path)
    monkeypatch.setattr(tasks, "official_tabx_root", lambda: tmp_path / "missing")
    with pytest.raises(ImportError, match="git submodule update --init --recursive"):
        import_tabx()


def test_json_parameters_and_output_root_are_part_of_the_contract(tmp_path):
    identity = simulator_identity()
    assert identity["submodule_commit"]
    assert len(identity["parameter_json_sha256"]) == 2
    assert "src/tabx/heuristic_policy/parameters/advanced.json" in identity["parameter_json_sha256"]
    for path, digest in identity["parameter_json_sha256"].items():
        assert file_sha256(official_tabx_root() / path) == digest
    old = tmp_path / "old"
    old.mkdir()
    (old / "eval_queue.sqlite3").touch()
    with pytest.raises(ValueError, match="new output directory"):
        ensure_output_contract(old)
    new = tmp_path / "new"
    ensure_output_contract(new)
    assert read_json(new / "tabx_semantics.json") == identity
    atomic_json(new / "tabx_semantics.json", dict(identity, layout_version="legacy"))
    with pytest.raises(ValueError, match="different TABX"):
        ensure_output_contract(new)

    old_parameters = dict(identity["parameter_json_sha256"])
    old_parameters.pop("src/tabx/heuristic_policy/parameters/advanced.json")
    medium_path = "src/tabx/heuristic_policy/parameters/medium.json"
    old_parameters[medium_path] = file_sha256(official_tabx_root() / medium_path)
    marker = new / "tabx_semantics.json"
    atomic_json(marker, dict(identity, parameter_json_sha256=old_parameters))
    original = marker.read_bytes()
    with pytest.raises(ValueError, match="different TABX"):
        ensure_output_contract(new)
    assert marker.read_bytes() == original


def test_native_opponent_uses_official_advanced_parameters():
    from envs.tabx_tasks import build_task_params
    params, _ = build_task_params(tiny_task())
    assert float(np.asarray(params["heuristic_params"].epsilon).item()) == pytest.approx(.1)
    assert float(np.asarray(params["heuristic_params"].aggressive_threshold).item()) == pytest.approx(.5)


def test_parameter_json_drift_rejects_a_snapshot(monkeypatch, tmp_path):
    import tabx_storage
    provenance = dict(epymarl=dict(source_hash="same"), tabx={**simulator_identity(), "source_hash": "same"},
                      versions={name: "1.0" for name in ("jax", "jaxlib", "torch")})
    identity = dict(task_id="23", task_hash="task", bank_hash="bank")
    (tmp_path / "agent.th").write_bytes(b"weights")
    atomic_json(tmp_path / "config.json", {})
    atomic_json(tmp_path / "manifest.json", dict(identity, timeout_rule="hp", simulator_contract=SIMULATOR_CONTRACT,
                layout_version=LAYOUT_VERSION, agent_sha256=file_sha256(tmp_path / "agent.th"),
                config_sha256=file_sha256(tmp_path / "config.json"), provenance=provenance))
    current = deepcopy(provenance)
    current["tabx"]["parameter_json_sha256"] = {"changed.json": "different"}
    monkeypatch.setattr(tabx_storage, "provenance", lambda: current)
    with pytest.raises(ValueError, match="parameter_json_sha256"):
        tabx_storage.validate_snapshot(tmp_path, identity, "hp")


@pytest.mark.parametrize("rule,win", [("hp", True), ("loss", False)])
def test_sub_0001_hp_lead_is_compared_without_rounding(rule, win):
    assert round(0.50004, 4) == round(0.5, 4)
    outcome = adjudicate(True, True, 0.50004, 0.5, 512, rule=rule)
    assert outcome.is_win == win
    assert replace_terminal_reward(-0.75, -1., outcome)[0] == (1.25 if win else -0.75)


def native_checks():
    """Run in a fresh interpreter so the official src namespace cannot be preloaded."""
    from envs.tabx_tasks import load_task, build_task_params
    from envs.tabx_wrapper import TABXWrapper
    import jax
    for key, sizes in (("23", (2, 3, 0)), ("647", (10, 10, 2)), ("657", (4, 4, 10)), ("651", (5, 5, 5))):
        task, _ = load_task(os.environ["TABX_BANK"], key)
        original = deepcopy(task)
        params, cfg = build_task_params(task)
        assert (cfg.max_n_ally, cfg.max_n_enemy, cfg.max_n_zone) == sizes
        assert task == original
        for name, value in task["scenario"].items():
            np.testing.assert_array_equal(getattr(params["scenario"], name), np.asarray(value, dtype=np.asarray(getattr(params["scenario"], name)).dtype))
        assert params["zone_scenario"].position.shape == (sizes[2], 2)
        assert int(np.asarray(params["zone_scenario"].n_zone).item()) == sizes[2]
        env = TABXWrapper(key, task_bank=os.environ["TABX_BANK"])
        env.reset(seed=7)
        native_key, reset_key = jax.random.split(jax.random.PRNGKey(7))
        native_obs, native_state = env._reset(reset_key, env.params)
        rng = np.random.default_rng(71)
        for _ in range(12):
            for a, b in zip(jax.tree.leaves(native_state), jax.tree.leaves(env.env_state)):
                np.testing.assert_array_equal(a, b)
            for observer, name in enumerate(env.base_env.ally_keys):
                np.testing.assert_array_equal(env.layout.unpack_local(env.get_obs()[observer], observer), native_obs[name])
            np.testing.assert_array_equal(env.layout.unpack_state(env.get_state()), native_obs["world_state"])
            assert np.isfinite(env.get_obs()).all() and np.isfinite(env.get_state()).all()
            np.testing.assert_array_equal(env.get_frame()["present"], env.layout.present[:10])
            native_avail = env.simulator.get_avail_actions(native_state)
            expected_avail = env.layout.pack_avail_actions([native_avail[k] for k in env.base_env.ally_keys], env._native_alive[:sizes[0]])
            np.testing.assert_array_equal(env.get_avail_actions(), expected_avail)
            action = np.array([rng.choice(np.flatnonzero(row)) for row in expected_avail])
            native_key, step_key = jax.random.split(native_key)
            native_obs, native_state, rewards, _, info = env._step(step_key, native_state,
                {name: jax.numpy.asarray(action[i]) for i, name in enumerate(env.base_env.ally_keys)})
            _, _, ended, _, _ = env.step(action)
            assert env.last_dense_reward == float(np.asarray(rewards["__all__"]).item()) - float(np.asarray(info["done_reward"])[0])
            if ended:
                break
        env.close()
    # Force the exact boundary using real units and the original alive cache.
    for rule in ("hp", "loss"):
        for ally_hp, enemy_hp, elimination in ((0.8, 0.5, False), (0.5, 0.5, False), (0.2, 0.5, False),
                                                (0.50004, 0.5, False), (1., 0., True)):
            env = TABXWrapper("23", task_bank=os.environ["TABX_BANK"], timeout_rule=rule)
            env.reset(seed=7)
            state = dict(env.env_state["state"])
            for i, name in enumerate(env.base_env.unit_keys):
                unit = state[name]
                hp = ally_hp if i < env.layout.n_ally else enemy_hp
                state[name] = unit.replace(status=unit.status.replace(health=unit.status.max_health * hp))
            state["game_manager"] = state["game_manager"].replace(timestep=jax.numpy.array([511]))
            env.env_state = dict(env.env_state, state=state)
            if ally_hp == 0.50004:
                reference = env._step(jax.random.split(env._episode_key)[1], deepcopy(env.env_state),
                    {name: jax.numpy.asarray(7) for name in env.base_env.ally_keys})
                assert float(np.asarray(reference[4]["done_reward"])[0]) == -1.
            _, reward, ended, _, info = env.step(np.full(10, 7))
            assert ended and not info["episode_limit"] and info["hit_time_limit"]
            assert info["end_reason"] == (1 if elimination else 2)
            expected_win = elimination or (rule == "hp" and ally_hp > enemy_hp)
            assert bool(info["is_win"]) == expected_win
            assert reward == env.last_dense_reward + (1. if expected_win else -1.)
            env.close()
