"""Opt-in real official TABX / QMIX two-task continuation acceptance."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tabx_storage import atomic_json, read_json, completed_checkpoints, validate_parent_checkpoint
from test_tabx_official import tiny_task


DELAYED_WORKER = r'''
import sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import tabx_pipeline as pipeline
from tabx_storage import read_json
original=pipeline.process_request
def delayed(queue,claim,bank,tabx_root=None):
    request=claim[2]
    if request.get('opponent_difficulty')=='medium':
        marker=queue.root/'advanced-started'/request['task_id']
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            if marker.exists() and (Path(marker.read_text())/'manifest.json').exists():
                break
            time.sleep(.05)
        else:
            raise RuntimeError('Advanced did not start while medium evaluation waited')
    return original(queue,claim,bank,tabx_root)
pipeline.process_request=delayed
pipeline.main(sys.argv[2:])
'''


@pytest.mark.skipif(os.environ.get('TABX_RUN_CURRICULUM_SMOKE') != '1',
                    reason='Opt in with TABX_RUN_CURRICULUM_SMOKE=1')
def test_real_two_tasks_final_checkpoint_handoff_and_delayed_medium_evaluation(tmp_path, monkeypatch):
    import tabx_scheduler as scheduler
    from tabx_pipeline import parser
    for name in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES'):
        monkeypatch.delenv(name, raising=False)
    tasks = []
    for key in ('23', '647'):
        task = tiny_task()
        task['task_id'] = key
        # Keep the short acceptance episodes alive through the timeout limit.
        task['scenario']['positions'] = [[-8., 0.], [8., 0.]]
        task['scenario']['healths'] = [[100.], [100.]]
        tasks.append(task)
    bank = Path(os.environ['TABX_BANK']) if os.environ.get('TABX_BANK') else tmp_path / 'two-tasks.json'
    if not os.environ.get('TABX_BANK'):
        atomic_json(bank, dict(schema_version='1.0', tasks=tasks))
    monkeypatch.setattr(scheduler, 'probe_gpus', lambda gpus: {gpu: dict(name='CPU acceptance only') for gpu in gpus})
    original_command = scheduler.BatchScheduler.training_command
    def command(self, job, attempt):
        actual = original_command(self, job, attempt)
        if attempt['stage'] == 'advanced':
            marker = self.root / 'advanced-started' / job['task_id']
            marker.parent.mkdir(exist_ok=True)
            marker.write_text(str(self.root / attempt['run']))
        # Different complete episode sizes produce different actual handoff steps.
        actual.append('env_args.max_episode_steps=' + ('8' if job['task_id'] == '23' else '6'))
        return ['use_cuda=False' if part == 'use_cuda=True' else
                'tabx_require_cuda=False' if part == 'tabx_require_cuda=True' else part for part in actual]
    monkeypatch.setattr(scheduler.BatchScheduler, 'training_command', command)
    def workers(self):
        actual = [sys.executable, '-c', DELAYED_WORKER, str(scheduler.REPOSITORY / 'src'),
            'eval-worker', '--root', str(self.root), '--bank', str(bank), '--rule', 'hp',
            '--batch-id', self.args.batch_id, '--workers', '1', '--external-dispatch', '--poll-seconds', '.05']
        self.worker_log = (self.directory / 'eval-workers.log').open('a')
        self.worker = subprocess.Popen(actual, cwd=scheduler.REPOSITORY, start_new_session=True,
            stdout=self.worker_log, stderr=subprocess.STDOUT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu'))
        self.state['worker'] = dict(pid=self.worker.pid, process_start=scheduler.process_identity(self.worker.pid))
    monkeypatch.setattr(scheduler.BatchScheduler, 'start_workers', workers)
    root = tmp_path / 'output'
    argv = ['batch-train', '--root', str(root), '--bank', str(bank), '--rule', 'hp', '--tasks', '23,647',
            '--seeds', '1', '--gpus', '0,1', '--batch-id', 'curriculum-smoke', '--eval-workers', '1',
            '--poll-seconds', '.05', '--medium-steps', '9', '--advanced-steps', '9']
    for setting in ('batch_size=2', 'buffer_size=4', 'save_model_interval=8', 'tabx_screen_episodes=2',
                    'learner_log_interval=8', 'epsilon_anneal_time=80'):
        argv.extend(['--set', setting])
    assert scheduler.batch_train(parser().parse_args(argv)) == 0
    directory = root / '_batches/hp/curriculum-smoke'
    state = read_json(directory / 'state.json')
    audit = []
    for job in state['jobs']:
        medium, advanced = job['attempts']
        parent = job['parent_checkpoint']
        medium_run, advanced_run = root / medium['run'], root / advanced['run']
        parent_manifest = validate_parent_checkpoint(root, parent, read_json(medium_run / 'manifest.json'), 1, 'hp')
        assert parent['run_id'] == medium['run_id']
        assert parent['task_id'] == job['task_id']
        assert parent['t_env'] == completed_checkpoints(medium_run)[0]['final_t_env']
        child_config = read_json(advanced_run / 'config.json')
        assert child_config['t_max'] == parent['t_env'] + 9
        assert child_config['checkpoint_path'] == str(root / parent['snapshot'])
        assert read_json(advanced_run / 'manifest.json')['parent_checkpoint'] == parent
        assert completed_checkpoints(advanced_run)[0]['final_t_env'] >= parent['t_env'] + 9
        assert completed_checkpoints(advanced_run)[0]['learner_updates'] > 0
        episodes = [__import__('json').loads(line) for line in (advanced_run / 'training_episodes.jsonl').read_text().splitlines()]
        assert episodes[0]['episode_index'] == parent_manifest['training_episodes']
        # First advanced episode explores at the parent's actual global step.
        assert episodes[0]['epsilon'] == pytest.approx(max(.05, 1 - parent['t_env'] * .95 / 80))
        delayed = False
        for attempt in (medium, advanced):
            for result in (root / attempt['run'] / 'evaluations').glob('*.json'):
                report = read_json(result)
                assert report['opponent_difficulty'] == attempt['stage']
                assert report['request']['opponent_difficulty'] == attempt['stage']
                if attempt is medium:
                    assert report['started_at'] > advanced['started_at']
                    delayed = True
        assert delayed and job['status'] == 'complete'
        audit.append(dict(task_id=job['task_id'], medium_final_step=parent['t_env'],
                          medium_run_id=medium['run_id'], advanced_run_id=advanced['run_id'],
                          advanced_t_max=child_config['t_max'], parent_checkpoint=parent['snapshot'],
                          delayed_medium_evaluation='passed'))
    assert audit[0]['medium_final_step'] != audit[1]['medium_final_step']
    assert scheduler.batch_train(parser().parse_args(argv + ['--resume'])) == 0
    assert [len(job['attempts']) for job in read_json(directory / 'state.json')['jobs']] == [2, 2]
    # The same real advanced checkpoint must create an advanced environment at
    # all ordinary entry points before a learner or replay exporter is involved.
    from copy import deepcopy
    import logging
    import run as training
    from envs.tabx_wrapper import TABXWrapper
    first = read_json(directory / 'state.json')['jobs'][0]
    advanced_run = root / first['attempts'][-1]['run']
    advanced_snapshot = completed_checkpoints(advanced_run)[2][-1]
    template = read_json(advanced_snapshot / 'config.json')
    original_init = TABXWrapper.__init__
    observed = []
    class CreatedEnvironment(Exception):
        pass
    def observe_environment(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        observed.append(self.opponent_difficulty)
        assert self.opponent_difficulty == 'advanced'
        raise CreatedEnvironment
    with monkeypatch.context() as check:
        check.setattr(TABXWrapper, '__init__', observe_environment)
        check.setattr(training.Logger, 'setup_sacred', lambda *args: None)
        for mode in ('train', 'evaluate', 'save_replay'):
            cfg = deepcopy(template)
            cfg.update(checkpoint_path=str(advanced_snapshot.parent), tabx_parent_checkpoint=None,
                       tabx_run_id='ordinary-' + mode, tabx_batch_id=None, tabx_train_log=None,
                       evaluate=mode == 'evaluate', save_replay=mode == 'save_replay')
            cfg['env_args']['opponent_difficulty'] = None
            with pytest.raises(CreatedEnvironment):
                training.run(None, cfg, logging.getLogger('restore-acceptance'))
            saved = read_json(root / 'hp/23/1' / ('ordinary-' + mode) / 'manifest.json')
            assert saved['opponent_difficulty'] == 'advanced'
            assert saved['difficulty_resolution']['checkpoint'] == str(advanced_snapshot)
            assert saved['difficulty_resolution']['source'] == 'checkpoint'
    assert observed == ['advanced'] * 3
    atomic_json(tmp_path / 'acceptance.json', dict(status='passed', tasks=audit,
                ordinary_restore_environment_difficulties=observed))
