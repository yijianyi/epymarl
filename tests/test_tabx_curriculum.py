"""Two-stage orchestration with real subprocesses and immutable inventories."""
import json
import sys

import pytest

import tabx_scheduler as scheduler
from tabx_pipeline import parser
from tabx_reporting import read_csv
from tabx_storage import read_json, final_checkpoint_identity
from test_tabx_scheduler import batch_fixture


TRAIN = r'''
import json,os,sys,time
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0,sys.argv[1])
from tabx_storage import SnapshotPublisher,atomic_json,read_json
job=json.loads(sys.argv[2]);run=Path(job['run']);run.mkdir(parents=True)
parent=job.get('parent');stage=job['stage'];start=parent['t_env'] if parent else 0
atomic_json(run/'manifest.json',dict(task_id=job['task_id'],train_seed=job['seed'],timeout_rule=job['rule'],
    bank_hash=job['bank_hash'],task_hash=job['task_hash'],batch_id=job['batch_id'],run_id=job['run_id'],master_seed=0,
    provenance={},native_shape=[1,1,0],opponent_difficulty=stage,parent_checkpoint=parent))
atomic_json(run/'config.json',job)
(run/'train.log').symlink_to(os.path.relpath(job['log'],run))
if job.get('fail'):
    sys.exit(5)
args=NS(tabx_output_root=job['root'],tabx_run_dir=str(run),save_model_interval=8,tabx_screen_episodes=2,
    env_args=dict(opponent_difficulty=stage))
class Learner:
    training_steps=0
    def save_models(self,path):
        for name in ('agent.th','mixer.th','opt.th'):
            (Path(path)/name).write_bytes((job['task_id']+str(job['seed'])+str(self.training_steps)).encode())
learner=Learner();publisher=SnapshotPublisher(args,None)
if not parent:
    publisher.publish(learner,0)
else:
    assert read_json(Path(job['root'])/parent['snapshot']/'manifest.json')['run_id']==parent['run_id']
    publisher.next_save=(start//8+1)*8
# Different tasks end medium at different actual steps and wall times.
time.sleep(.12 if job['task_id']=='23' else .35)
learner.training_steps+=1
publisher.publish(learner,start+8)
time.sleep(.12 if job['task_id']=='23' else .35)
final=start+(16 if stage=='advanced' or job['task_id']=='23' else 24)
learner.training_steps+=1
# Complete must publish the non-periodic final model itself.
publisher.complete(learner,final,dict(t_env=final,learner_updates=2,elapsed_seconds=.4))
if job.get('corrupt'):
    (run/'checkpoints'/str(final)/'opt.th').write_bytes(b'corrupted')
'''


@pytest.fixture
def curriculum(batch_fixture, monkeypatch):
    f = batch_fixture
    f.argv += ['--medium-steps', '15', '--advanced-steps', '13']
    f.argv[f.argv.index('23,651,647')] = '23,651'
    f.fail_advanced, f.corrupt_medium = set(), set()
    # Preserve the production command builder for assertions, before substituting
    # lightweight checkpoint producers for GPU training.
    def command(self, job, attempt):
        stage = attempt['stage']
        data = dict(run=str(self.root / attempt['run']), log=str(self.root / attempt['log']), root=str(self.root),
                    run_id=attempt['run_id'], task_id=job['task_id'], seed=job['train_seed'], rule=self.args.rule,
                    bank_hash=self.manifest['bank_hash'], task_hash=self.manifest['task_hashes'][job['task_id']],
                    batch_id=self.args.batch_id, stage=stage,
                    parent=job.get('parent_checkpoint') if stage == 'advanced' else None,
                    fail=stage == 'advanced' and job['task_id'] in f.fail_advanced,
                    corrupt=stage == 'medium' and job['task_id'] in f.corrupt_medium)
        actual = f.production_command(self, job, attempt)
        assert f'env_args.opponent_difficulty="{stage}"' in actual
        endpoint = 15 if stage == 'medium' else data['parent']['t_env'] + 13
        assert f't_max={endpoint}' in actual
        if stage == 'advanced':
            assert f'checkpoint_path="{self.root / data["parent"]["snapshot"]}"' in actual
        return [sys.executable, '-c', TRAIN, str(scheduler.REPOSITORY / 'src'), json.dumps(data)]
    monkeypatch.setattr(scheduler.BatchScheduler, 'training_command', command)
    return f


def test_own_final_model_zero_win_handoff_delayed_medium_eval_and_resume(curriculum):
    f = curriculum
    assert scheduler.batch_train(parser().parse_args(f.argv)) == 0
    state = read_json(f.directory / 'state.json')
    assert state['status'] == 'complete'
    for job in state['jobs']:
        medium, advanced = job['attempts']
        assert [medium['stage'], advanced['stage']] == ['medium', 'advanced']
        parent = job['parent_checkpoint']
        assert parent == final_checkpoint_identity(f.root, f.root / medium['run'])
        assert parent['t_env'] == (16 if job['task_id'] == '23' else 24)
        assert parent['train_seed'] == job['train_seed'] and parent['task_id'] == job['task_id']
        assert advanced['parent_checkpoint'] == parent
        assert advanced['run'] != medium['run']
        assert job['train_status'] == job['eval_status'] == 'complete'
        assert job['eval_done'] == job['eval_expected'] == 5
        for attempt in (medium, advanced):
            run = f.root / attempt['run']
            rows = read_csv(run / 'eval_summary.csv')
            assert {row['opponent_difficulty'] for row in rows} == {attempt['stage']}
            assert {float(row['win_rate_pct']) for row in rows} == {0.0}
        reports = [read_json(p) for p in (f.root / medium['run'] / 'evaluations').glob('*.json')]
        # A medium result actually executed after this task started advanced.
        assert any(report['evaluated_at'] > advanced['started_at'] for report in reports)
        assert all(report['opponent_difficulty'] == 'medium' for report in reports)
    first, second = state['jobs']
    assert first['attempts'][1]['started_at'] < second['attempts'][0]['ended_at']
    assert scheduler.batch_train(parser().parse_args(f.argv + ['--resume'])) == 0
    after = read_json(f.directory / 'state.json')
    assert [len(job['attempts']) for job in after['jobs']] == [2, 2]
    assert [job['parent_checkpoint'] for job in after['jobs']] == [job['parent_checkpoint'] for job in state['jobs']]


def test_resume_between_handoff_and_child_launch(curriculum, monkeypatch):
    f = curriculum
    original = scheduler.BatchScheduler.plan_next_stage
    def stop_after_handoff(self, job, **kwargs):
        original(self, job, **kwargs)
        if job.get('parent_checkpoint'):
            self.stopping = True
    monkeypatch.setattr(scheduler.BatchScheduler, 'plan_next_stage', stop_after_handoff)
    assert scheduler.batch_train(parser().parse_args(f.argv)) == 130
    before = read_json(f.directory / 'state.json')
    parent = before['jobs'][0]['parent_checkpoint']
    assert len(before['jobs'][0]['attempts']) == 1
    monkeypatch.setattr(scheduler.BatchScheduler, 'plan_next_stage', original)
    assert scheduler.batch_train(parser().parse_args(f.argv + ['--resume'])) == 0
    after = read_json(f.directory / 'state.json')
    assert after['jobs'][0]['parent_checkpoint'] == parent
    assert len(after['jobs'][0]['attempts']) == 2


def test_failed_advanced_retries_same_parent_without_retraining_medium(curriculum):
    f = curriculum
    f.fail_advanced.add('651')
    assert scheduler.batch_train(parser().parse_args(f.argv)) == 1
    before = read_json(f.directory / 'state.json')
    assert before['jobs'][0]['status'] == 'complete'
    parent = before['jobs'][1]['parent_checkpoint']
    f.fail_advanced.clear()
    assert scheduler.batch_train(parser().parse_args(f.argv + ['--resume'])) == 0
    after = read_json(f.directory / 'state.json')
    assert [attempt['stage'] for attempt in after['jobs'][1]['attempts']] == ['medium', 'advanced', 'advanced']
    assert after['jobs'][1]['parent_checkpoint'] == parent


def test_bad_final_checkpoint_fails_only_its_handoff(curriculum):
    f = curriculum
    f.corrupt_medium.add('651')
    assert scheduler.batch_train(parser().parse_args(f.argv)) == 1
    jobs = read_json(f.directory / 'state.json')['jobs']
    assert jobs[0]['status'] == 'complete'
    assert jobs[1]['status'] == 'handoff_failed'
    assert len(jobs[1]['attempts']) == 1
    assert 'opt.th' in jobs[1]['handoff_error'] and 'training_complete.json' in jobs[1]['handoff_error']


@pytest.mark.parametrize('options', [
    ['--medium-steps', '10'], ['--advanced-steps', '10'],
    ['--medium-steps', '0', '--advanced-steps', '10'],
    ['--medium-steps', '10', '--advanced-steps', '-1'],
    ['--medium-steps', '10', '--advanced-steps', '10', '--set', 't_max=10'],
])
def test_stage_budget_validation(batch_fixture, options):
    with pytest.raises(ValueError):
        scheduler.batch_train(parser().parse_args(batch_fixture.argv + options))


def test_each_seed_has_its_own_parent(curriculum):
    f = curriculum
    assert scheduler.batch_train(parser().parse_args(f.argv + ['--seeds', '1,2'])) == 0
    jobs = read_json(f.directory / 'state.json')['jobs']
    assert len(jobs) == 4
    for job in jobs:
        parent = job['parent_checkpoint']
        assert (parent['task_id'], parent['train_seed']) == (job['task_id'], job['train_seed'])
        assert f"/{job['train_seed']}/" in parent['snapshot']
    assert len({job['parent_checkpoint']['snapshot'] for job in jobs}) == 4


def test_corrupt_completion_stays_isolated_on_resume(curriculum):
    f = curriculum
    f.corrupt_medium.add('651')
    assert scheduler.batch_train(parser().parse_args(f.argv)) == 1
    before = read_json(f.directory / 'state.json')
    assert scheduler.batch_train(parser().parse_args(f.argv + ['--resume'])) == 1
    after = read_json(f.directory / 'state.json')
    assert after['jobs'][0]['status'] == 'complete'
    assert after['jobs'][1]['status'] == 'handoff_failed'
    assert [len(job['attempts']) for job in after['jobs']] == [len(job['attempts']) for job in before['jobs']]


def test_result_must_report_the_checkpoint_difficulty(tmp_path):
    from tabx_storage import EvalQueue
    queue = EvalQueue(tmp_path)
    request = dict(rule='hp', seeds=[1], checkpoint_hash='weights', opponent_difficulty='medium')
    key = queue.enqueue(request)
    _, owner, _ = queue.claim('hp')
    queue.record(key, owner, dict(seed=1, return_value=0))
    report = dict(request=request, episodes=queue.results(key), opponent_difficulty='advanced')
    with pytest.raises(ValueError, match='opponent difficulty mismatch'):
        queue.publish_result(key, owner, report)
    report['opponent_difficulty'] = 'medium'
    queue.publish_result(key, owner, report)
    queue.finish(key, owner)
    assert queue.completed_result(key)['opponent_difficulty'] == 'medium'
