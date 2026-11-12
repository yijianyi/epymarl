"""Plot the HP batch, preserving source files and validating exports against logs."""
import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/epymarl-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

TASKS = ('657', '647', '403')
METRICS = ('loss', 'win_rate', 'reward')
TITLES = ('训练 loss', '评估胜率', '评估 reward')
COLORS = ('#2563A6', '#138875', '#D47C21')


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def column(rows, key):
    return np.asarray([r[key] for r in rows], dtype=float)


def rolling(values, window=21):
    values = np.asarray(values, dtype=float)
    sums = np.r_[0., np.cumsum(values)]
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    return (sums[end] - sums[start]) / (end - start)


def load(source):
    data = defaultdict(lambda: {'train': [], 'eval': [], 'episodes': []})
    for line in (source / '_batch/batch.log').open():
        match = re.match(r'\[(TRAIN|EVAL)\] (.*)', line)
        if not match:
            continue
        d = dict(re.findall(r'(\w+)=(\S+)', match[2]))
        assert d['rule'] == 'hp' and d['seed'] == '1', d
        if match[1] == 'TRAIN':
            data[d['task']]['train'].append(dict(step=int(d['step']), loss=float(d['loss'])))
        else:
            assert d['phase'] == 'screen'
            data[d['task']]['eval'].append(dict(
                step=int(d['checkpoint_step']), episodes=int(d['episodes']), wins=int(d['wins']),
                reward=float(d['mean_return'])))
    for task, run in data.items():
        folder = source / task
        if (folder / 'train_metrics.csv').exists():
            rows = read_csv(folder / 'train_metrics.csv')
            assert len(rows) == len(run['train'])
            exported = [dict(step=int(d['t_env']), loss=float(d['loss'])) for d in rows]
            logged = {d['step']: d for d in run['train']}
            for d in exported:
                assert np.isclose(d['loss'], logged[d['step']]['loss'], rtol=6e-6, atol=1e-9)
            run['train'] = exported
        if (folder / 'eval_summary.csv').exists():
            rows = read_csv(folder / 'eval_summary.csv')
            assert len(rows) == len(run['eval'])
            logged = {d['step']: d for d in run['eval']}
            exported = []
            for d in rows:
                assert d['phase'] == 'screen' and d['action_selection'] == 'greedy'
                row = dict(step=int(d['checkpoint_step']), episodes=int(d['episodes']),
                           wins=int(d['wins']), reward=float(d['mean_return']))
                old = logged[row['step']]
                assert row['wins'] == old['wins'] and row['episodes'] == old['episodes']
                assert np.isclose(row['reward'], old['reward'], rtol=6e-6, atol=1e-6)
                exported.append(row)
            run['eval'] = exported
        if (folder / 'training_episodes.jsonl').exists():
            run['episodes'] = [json.loads(s) for s in (folder / 'training_episodes.jsonl').read_text().splitlines() if s.strip()]
        for key in ('train', 'eval'):
            run[key].sort(key=lambda d: d['step'])
            steps = column(run[key], 'step')
            assert np.all(np.diff(steps) > 0)
            assert np.all(np.isfinite(column(run[key], 'loss' if key == 'train' else 'reward')))
        assert np.all(column(run['train'], 'loss') > 0)
        assert len(run['eval']) == 61
        for d in run['eval']:
            assert d['episodes'] == 32 and 0 <= d['wins'] <= d['episodes']
            d['win_rate'] = 100 * d['wins'] / d['episodes']
    summary = read_csv(source / '_batch/batch_summary.csv')
    assert set(data) == {d['task_id'] for d in summary} and len(data) == 32
    for d in summary:
        final = data[d['task_id']]['eval'][-1]
        assert d['status'] == 'complete' and final['step'] == int(d['latest_eval_step'])
        assert final['win_rate'] == float(d['latest_eval_win_rate_pct'])
        assert abs(final['reward'] - float(d['latest_eval_mean_return'])) < 1e-5
    return data


def decorate(ax, metric):
    ax.set_xlabel('环境步数（百万）')
    ax.set_xlim(0, 1.501)
    ax.set_xticks(np.arange(0, 1.51, .3))
    ax.grid(alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
    if metric == 'loss':
        ax.set_yscale('log')
        ax.set_ylabel('TD loss（对数刻度）')
    elif metric == 'win_rate':
        ax.set_ylabel('胜率（%）')
        ax.set_ylim(-3, 103)
    else:
        ax.set_ylabel('平均回合累计 reward')


def save(fig, out, name):
    fig.savefig(out / (name + '.png'), dpi=180, facecolor='white')
    fig.savefig(out / (name + '.svg'), facecolor='white')
    plt.close(fig)


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 11,
                         'axes.titlesize': 14, 'axes.unicode_minus': False, 'svg.fonttype': 'path'})
    data = load(args.source)
    aggregates = {}
    aggregate_rows = []
    for metric in METRICS:
        key = 'train' if metric == 'loss' else 'eval'
        runs = list(data.values())
        lo = max(r[key][0]['step'] for r in runs)
        hi = min(r[key][-1]['step'] for r in runs)
        grid = np.linspace(lo, hi, 601 if metric == 'loss' else 61)
        aligned = []
        for run in runs:
            y = column(run[key], metric)
            aligned.append(np.interp(grid, column(run[key], 'step'), rolling(y) if metric == 'loss' else y))
        mean = np.mean(aligned, axis=0)
        aggregates[metric] = grid, mean
        aggregate_rows.extend(dict(metric=metric, step=x, task_mean=y, n_tasks=32) for x, y in zip(grid, mean))

    note = '任务等权平均；按环境步数插值对齐，仅绘制公共观测区间。loss：21 点后向均值；评估：每点 32 局。'
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))
    fig.suptitle('HP · 全部 32 个任务的平均曲线', x=.04, ha='left', fontsize=20)
    for ax, metric, title, color in zip(axes, METRICS, TITLES, COLORS):
        grid, mean = aggregates[metric]
        ax.plot(grid / 1e6, mean, color=color, lw=2.3)
        ax.set_title(title)
        decorate(ax, metric)
    fig.text(.04, .025, note, fontsize=10, color='#555555')
    fig.tight_layout(rect=(.02, .08, .99, .87))
    save(fig, args.out, '01_overall_32_tasks')
    for metric, title, color in zip(METRICS, TITLES, COLORS):
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        grid, mean = aggregates[metric]
        ax.plot(grid / 1e6, mean, color=color, lw=2.3)
        ax.set_title('HP · 32 任务平均 · ' + title)
        decorate(ax, metric)
        fig.tight_layout()
        save(fig, args.out, 'overall_' + metric)

    def draw_tasks(tasks, name):
        fig, axes = plt.subplots(len(tasks), 3, figsize=(16.5, 3.6 * len(tasks) + 1.3), squeeze=False)
        fig.suptitle('HP · ' + ('任务 ' + tasks[0] if len(tasks) == 1 else '657 / 647 / 403 任务对比'),
                     x=.04, ha='left', fontsize=20)
        for task, row in zip(tasks, axes):
            for ax, metric, title, color in zip(row, METRICS, TITLES, COLORS):
                rows = data[task]['train' if metric == 'loss' else 'eval']
                x, y = column(rows, 'step') / 1e6, column(rows, metric)
                if metric == 'loss':
                    ax.plot(x, y, color=color, alpha=.2, lw=.7, label='原始值')
                    ax.plot(x, rolling(y), color=color, lw=2, label='21 点后向均值')
                    ax.legend(fontsize=9, frameon=False)
                else:
                    ax.plot(x, y, color=color, lw=1.8, marker='o', markersize=2.7)
                ax.set_title(task + ' · ' + title)
                decorate(ax, metric)
        fig.text(.04, .025 if len(tasks) == 1 else .017,
                 'loss：浅线为原始值，粗线为 21 点后向均值。胜率 / reward：greedy 评估原始值，每点 32 局；训练 seed=1。',
                 fontsize=10, color='#555555')
        fig.tight_layout(rect=(.02, .08 if len(tasks) == 1 else .05, .99, .87 if len(tasks) == 1 else .94))
        save(fig, args.out, name)

    for task in TASKS:
        draw_tasks((task,), 'task_' + task)
    draw_tasks(TASKS, '02_selected_tasks')

    fig, axes = plt.subplots(3, 2, figsize=(13, 11.5))
    fig.suptitle('HP · 训练回合胜率与 reward', x=.055, ha='left', fontsize=20)
    for task, row in zip(TASKS, axes):
        episodes = data[task]['episodes']
        for ax, field, metric, title, color in zip(row, ('is_win', 'return_value'),
                ('win_rate', 'reward'), ('训练胜率', '训练 reward'), COLORS[1:]):
            values = column(episodes, field) * (100 if field == 'is_win' else 1)
            ax.plot(column(episodes, 't_env') / 1e6, rolling(values, 100), color=color, lw=1.8)
            ax.set_title(task + ' · ' + title)
            decorate(ax, metric)
    fig.text(.055, .018, '最近 100 个训练回合的后向均值，包含 epsilon-greedy 探索；起始不足窗口时使用已有回合。', fontsize=10, color='#555555')
    fig.tight_layout(rect=(.025, .05, .99, .94))
    save(fig, args.out, '03_training_episode_curves')

    summaries = []
    for task in sorted(data, key=int):
        run = data[task]
        first, final = run['eval'][0], run['eval'][-1]
        summaries.append(dict(task_id=task, seed=1, train_points=len(run['train']), eval_points=len(run['eval']),
            final_step=final['step'], initial_win_rate_pct=first['win_rate'], final_win_rate_pct=final['win_rate'],
            final_reward=final['reward'], last5_win_rate_pct=float(np.mean(column(run['eval'][-5:], 'win_rate'))),
            last5_reward=float(np.mean(column(run['eval'][-5:], 'reward')))))
    write_csv(args.out / 'summary.csv', summaries)
    write_csv(args.out / 'overall_curves.csv', aggregate_rows)
    print(json.dumps({'tasks': len(data), 'train_points': sum(len(r['train']) for r in data.values()),
        'eval_points': sum(len(r['eval']) for r in data.values()), 'selected': [r for r in summaries if r['task_id'] in TASKS],
        'final_task_mean_win_rate_pct': float(np.mean([r['final_win_rate_pct'] for r in summaries])),
        'final_task_mean_reward': float(np.mean([r['final_reward'] for r in summaries]))}, indent=2))


if __name__ == '__main__':
    main()
