"""Restoration and historical reporting share evidence and reject contradictions."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from tabx_difficulty import (LEGACY_PRESETS, LEGACY_SIMULATOR_SOURCE_HASH, checkpoint_difficulty, evaluation_difficulty,
                             requested_difficulty, select_checkpoint)
from tabx_reporting import (evaluation_summary, read_csv, reconcile_eval_summary,
                            refresh_eval_summary, update_eval_summary)
from tabx_storage import atomic_json, file_sha256, prepare_tabx_run, read_json


def snapshot(path, difficulty=None, config_difficulty=None, **extra):
    path.mkdir(parents=True, exist_ok=True)
    (path / 'agent.th').write_bytes(b'agent')
    config = dict(env_args={})
    if config_difficulty:
        config['env_args']['opponent_difficulty'] = config_difficulty
    atomic_json(path / 'config.json', config)
    manifest = dict(t_env=int(path.name), run_id=path.parent.parent.name,
                    agent_sha256=file_sha256(path / 'agent.th'), config_sha256=file_sha256(path / 'config.json'), **extra)
    if difficulty:
        manifest['opponent_difficulty'] = difficulty
    atomic_json(path / 'manifest.json', manifest)
    return path


def config(path='', requested=None, **extra):
    return dict(env='tabx', checkpoint_path=str(path), load_step=0, seed=1,
                env_args=dict(key='23', timeout_rule='hp', opponent_difficulty=requested), **extra)


def test_select_once_before_run_and_runner_and_inherit_advanced_for_all_modes(tmp_path, monkeypatch):
    import run
    import tabx_storage
    import tabx_difficulty
    import jax
    path = snapshot(tmp_path / 'source/checkpoints/20', 'advanced', 'advanced')
    snapshot(path.parent / '10', 'medium', 'medium')
    original = tabx_difficulty.select_checkpoint
    selections = []
    def choose(*args):
        selected = original(*args)
        selections.append(selected)
        return selected
    monkeypatch.setattr(tabx_difficulty, 'select_checkpoint', choose)
    monkeypatch.setattr(run, 'args_sanity_check', lambda cfg, log: cfg)
    monkeypatch.setattr(jax.config, 'update', lambda *a: None)
    from envs import tabx_tasks
    monkeypatch.setattr(tabx_tasks, 'import_tabx', lambda *a: None)
    monkeypatch.setattr(tabx_tasks, 'load_task', lambda *a: ({}, dict(task_id='23', task_hash='t', bank_hash='b')))
    monkeypatch.setattr(tabx_tasks, 'validate_task_structure', lambda *a: (1, 1, 0))
    # Stop at the first runner construction: both persisted metadata and the
    # actual runner arguments must already contain the inherited difficulty.
    class ObservedRunner(Exception):
        pass
    captured = []
    from envs.tabx_wrapper import TABXWrapper
    def environment(self, **kwargs):
        captured.append((kwargs['opponent_difficulty'], str(path)))
        raise ObservedRunner
    monkeypatch.setattr(TABXWrapper, '__init__', environment)
    def create(cfg, manifest):
        assert cfg['env_args']['opponent_difficulty'] == 'advanced'
        assert cfg['checkpoint_path'] == str(path)
        directory = tmp_path / f"output/hp/23/1/{len(captured)}"
        directory.mkdir(parents=True)
        cfg['tabx_run_dir'] = str(directory)
        return directory
    monkeypatch.setattr(tabx_storage, 'create_run', create)
    monkeypatch.setattr(run.Logger, 'setup_sacred', lambda *a: None)
    logger = SimpleNamespace(info=lambda *a: None)
    for mode in ('train', 'evaluate', 'save_replay'):
        cfg = config(path.parent, use_cuda=False, runner='episode', batch_size_run=1, tabx_async_eval=False,
                     common_reward=True, reward_scalarisation='sum', learner='q_learner', evaluate=mode == 'evaluate', save_replay=mode == 'save_replay',
                     name='qmix', use_tensorboard=False, use_wandb=False)
        with pytest.raises(ObservedRunner):
            run.run(None, cfg, logger)
    assert captured == [('advanced', str(path))] * 3
    assert selections == [path] * 3


def test_conflict_fails_before_creating_a_run(tmp_path, monkeypatch):
    import run
    import tabx_storage
    from envs import tabx_tasks
    path = snapshot(tmp_path / 'source/checkpoints/20', 'advanced', 'advanced')
    monkeypatch.setattr(run, 'args_sanity_check', lambda cfg, log: cfg)
    monkeypatch.setattr(tabx_tasks, 'import_tabx', lambda *a: None)
    monkeypatch.setattr(tabx_tasks, 'load_task', lambda *a: ({}, {}))
    monkeypatch.setattr(tabx_tasks, 'validate_task_structure', lambda *a: (1, 1, 0))
    calls = []
    monkeypatch.setattr(tabx_storage, 'create_run', lambda *a: calls.append(a))
    with pytest.raises(ValueError) as error:
        run.run(None, config(path, 'medium', use_cuda=False, runner='episode', batch_size_run=1, tabx_async_eval=False), None)
    assert str(path) in str(error.value) and 'advanced' in str(error.value) and 'medium' in str(error.value)
    assert calls == []


def test_unknown_requires_explicit_launch_and_new_runs_default_to_medium(tmp_path):
    path = snapshot(tmp_path / 'source/checkpoints/20')
    with pytest.raises(ValueError, match='unknown.*explicitly'):
        prepare_tabx_run(config(path), {})
    cfg = prepare_tabx_run(config(path, 'advanced'), {})
    assert cfg['env_args']['opponent_difficulty'] == 'advanced'
    assert cfg['tabx_difficulty_resolution']['source'] == 'launch_argument'
    assert cfg['tabx_difficulty_resolution']['checkpoint_difficulty'] == 'unknown'
    assert prepare_tabx_run(config(), {})['env_args']['opponent_difficulty'] == 'medium'
    assert requested_difficulty(snapshot(path.parent / '30', 'advanced'), 'advanced')['difficulty'] == 'advanced'
    assert select_checkpoint(path.parent, 11) == path


def test_manifest_and_config_conflict_is_shared_by_restore_and_history(tmp_path):
    path = snapshot(tmp_path / 'hp/23/1/run/checkpoints/20', 'advanced', 'medium')
    with pytest.raises(ValueError, match='mismatch/conflict'):
        prepare_tabx_run(config(path), {})
    report = make_report(tmp_path, path)
    with pytest.raises(ValueError) as error:
        evaluation_summary('conflicting-row', report, 20, run_dir=path.parent.parent)
    assert 'conflicting-row' in str(error.value) and str(path) in str(error.value)
    assert 'advanced' in str(error.value) and 'medium' in str(error.value)


def make_report(root, path, difficulty=None, request_difficulty=None):
    request = dict(snapshot=str(path.relative_to(root)), run_id=path.parent.parent.name,
                   checkpoint_hash=file_sha256(path / 'agent.th'), seeds=[1, 2], phase='screen', action_selection='greedy')
    if request_difficulty:
        request['opponent_difficulty'] = request_difficulty
    report = dict(request=request, episodes=[dict(seed=1, is_win=0, return_value=-2., end_reason=2),
                                            dict(seed=2, is_win=1, return_value=3., end_reason=1)])
    if difficulty:
        report['opponent_difficulty'] = difficulty
    return report


def test_four_historical_cases_match_single_batch_and_full_rebuild(tmp_path):
    run = tmp_path / 'hp/23/1/run'
    reports = []
    for step, difficulty, legacy in ((10, 'medium', False), (20, 'advanced', False),
                                     (30, 'advanced', True), (40, None, True)):
        path = snapshot(run / 'checkpoints' / str(step), config_difficulty=difficulty)
        report = make_report(tmp_path, path, None if legacy else difficulty, None if legacy else difficulty)
        key = f'row-{step}'
        atomic_json(run / 'evaluations' / f'{key}.json', report)
        reports.append((key, step, report))
    originals = {p: p.read_bytes() for p in (run / 'evaluations').glob('*.json')}
    for key, step, report in reports:
        update_eval_summary(run, key, report, step)
    single = read_csv(run / 'eval_summary.csv')
    batch = {key: evaluation_summary(key, report, step, run_dir=run) for key, step, report in reports}
    reconcile_eval_summary(run, batch)
    assert read_csv(run / 'eval_summary.csv') == single
    refresh_eval_summary(run)
    assert read_csv(run / 'eval_summary.csv') == single
    assert [row['opponent_difficulty'] for row in single] == ['medium', 'advanced', 'advanced', 'unknown']
    assert {row['win_rate_pct'] for row in single} == {'50.0'}
    assert {row['mean_return'] for row in single} == {'0.5'}
    assert originals == {p: p.read_bytes() for p in originals}


def test_explicit_report_and_request_conflict_preserves_csv(tmp_path):
    run = tmp_path / 'hp/23/1/run'
    path = snapshot(run / 'checkpoints/10', 'medium')
    report = make_report(tmp_path, path, 'medium', 'advanced')
    run.mkdir(exist_ok=True)
    csv = run / 'eval_summary.csv'
    csv.write_bytes(b'original csv\n')
    with pytest.raises(ValueError) as error:
        update_eval_summary(run, 'bad-row', report, 10)
    assert 'bad-row' in str(error.value) and 'medium' in str(error.value) and 'advanced' in str(error.value)
    assert csv.read_bytes() == b'original csv\n'


def test_old_checkpoint_identity_must_match_its_evaluation(tmp_path):
    path = snapshot(tmp_path / 'hp/23/1/run/checkpoints/10', 'advanced')
    report = make_report(tmp_path, path)
    report['request']['run_id'] = 'other-run'
    with pytest.raises(ValueError, match='identity mismatch'):
        evaluation_difficulty(report, snapshot=path, checkpoint_step=10, request_id='wrong-model')


def test_provenance_requires_audited_fixed_preset_source_and_unique_parameters(tmp_path):
    path = snapshot(tmp_path / 'source/checkpoints/10')
    for source, (difficulty, digest) in LEGACY_PRESETS.items():
        provenance = dict(epymarl=dict(source_hash=source), tabx=dict(source_hash=LEGACY_SIMULATOR_SOURCE_HASH, parameter_json_sha256={
            f'src/tabx/heuristic_policy/parameters/{difficulty}.json': digest}))
        manifest = read_json(path / 'manifest.json')
        manifest['provenance'] = provenance
        atomic_json(path / 'manifest.json', manifest)
        assert checkpoint_difficulty(path)['difficulty'] == difficulty
        provenance['epymarl']['source_hash'] = 'unverified-source'
        atomic_json(path / 'manifest.json', manifest)
        assert checkpoint_difficulty(path)['difficulty'] == 'unknown'
        provenance['epymarl']['source_hash'] = source
        provenance['tabx']['parameter_json_sha256']['src/tabx/heuristic_policy/parameters/other.json'] = 'another'
        atomic_json(path / 'manifest.json', manifest)
        assert checkpoint_difficulty(path)['difficulty'] == 'unknown'


def test_config_only_checkpoint_and_missing_checkpoint_history(tmp_path):
    run = tmp_path / 'hp/23/1/run'
    path = snapshot(run / 'checkpoints/10', config_difficulty='advanced')
    report = make_report(tmp_path, path)
    (path / 'manifest.json').unlink()
    assert evaluation_summary('config-only', report, 10, run_dir=run)['opponent_difficulty'] == 'advanced'
    report['request']['snapshot'] = str(path)
    assert evaluation_summary('absolute-path', report, 10)['opponent_difficulty'] == 'advanced'
    report['request']['snapshot'] = 'hp/23/1/run/checkpoints/99'
    atomic_json(run / 'evaluations/missing.json', report)
    assert refresh_eval_summary(run)[0]['opponent_difficulty'] == 'unknown'


def test_new_reports_do_not_open_checkpoint_files(tmp_path, monkeypatch):
    run = tmp_path / 'hp/23/1/run'
    path = snapshot(run / 'checkpoints/10', 'advanced')
    report = make_report(tmp_path, path, 'advanced', 'advanced')
    original = Path.open
    def no_history(path, *args, **kwargs):
        if 'checkpoints' in path.parts:
            raise AssertionError('New reports must use actual evaluation metadata')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', no_history)
    assert update_eval_summary(run, 'new-report', report, 10)['opponent_difficulty'] == 'advanced'
