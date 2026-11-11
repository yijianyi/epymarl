"""Reproduce comparisons from read-only TABX batch logs and per-task exports."""
import argparse
import csv
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'epymarl-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

RULES = ('hp', 'loss')
TASKS = ('657', '647', '403')
COLORS = {'hp': '#2463B4', 'loss': '#DF7626'}
STYLES = {'hp': '-', 'loss': '--'}
LIMIT = 1_500_000


def csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def load(root):
    data = {r: defaultdict(lambda: {'train': [], 'eval': [], 'episodes': []}) for r in RULES}
    for rule in RULES:
        for line in (root / rule / '_batch/batch.log').open():
            match = re.match(r'\[(TRAIN|EVAL)\] (.*)', line)
            if not match:
                continue
            d = dict(re.findall(r'(\w+)=(\S+)', match[2]))
            if d['rule'] != rule or d['seed'] != '1':
                raise ValueError(f'Unexpected run: {line}')
            if match[1] == 'TRAIN':
                row = {'step': int(d['step']), 'loss': float(d['loss'])}
                key = 'train'
            else:
                if d['phase'] != 'screen':
                    raise ValueError(f'Unexpected evaluation phase: {line}')
                row = {'step': int(d['checkpoint_step']), 'episodes': int(d['episodes']),
                       'wins': int(d['wins']), 'return': float(d['mean_return']),
                       'elimination_wins': int(d['elimination_wins']),
                       'timeout_wins': int(d['timeout_wins'])}
                key = 'eval'
            data[rule][d['task']][key].append(row)
        for task, run in data[rule].items():
            folder = root / rule / task
            # Prefer full precision exports when available (the three requested tasks).
            if (folder / 'train_metrics.csv').exists():
                rows = csv_rows(folder / 'train_metrics.csv')
                assert len(rows) == len(run['train'])
                run['train'] = [{'step': int(d['t_env']), 'loss': float(d['loss'])} for d in rows]
            if (folder / 'eval_summary.csv').exists():
                rows = csv_rows(folder / 'eval_summary.csv')
                assert len(rows) == len(run['eval'])
                assert all(d['phase'] == 'screen' and d['action_selection'] == 'greedy' for d in rows)
                run['eval'] = [{'step': int(d['checkpoint_step']), 'episodes': int(d['episodes']),
                                'wins': int(d['wins']), 'return': float(d['mean_return']),
                                'elimination_wins': int(d['elimination_wins']),
                                'timeout_wins': int(d['timeout_wins'])} for d in rows]
            if (folder / 'training_episodes.jsonl').exists():
                run['episodes'] = [json.loads(line) for line in (folder / 'training_episodes.jsonl').open()]
            for key in ('train', 'eval'):
                run[key].sort(key=lambda d: d['step'])
                steps = [d['step'] for d in run[key]]
                assert len(steps) == len(set(steps)), (rule, task, key, 'duplicate steps')
                assert all(np.isfinite(d.get('loss', d.get('return'))) for d in run[key])
            assert len(run['eval']) == 61, (rule, task)
            assert all(d['episodes'] == 32 and d['wins'] == d['elimination_wins'] + d['timeout_wins'] for d in run['eval'])
            for d in run['eval']:
                d['win_rate'] = 100 * d['wins'] / d['episodes']
                d['elimination_rate'] = 100 * d['elimination_wins'] / d['episodes']
                d['timeout_rate'] = 100 * d['timeout_wins'] / d['episodes']
        summary = csv_rows(root / rule / '_batch/batch_summary.csv')
        assert len(data[rule]) == len(summary) == 32
        for row in summary:
            run = data[rule][row['task_id']]
            assert row['status'] == 'complete'
            assert run['eval'][-1]['step'] == int(row['latest_eval_step'])
            assert abs(run['eval'][-1]['win_rate'] - float(row['latest_eval_win_rate_pct'])) < 1e-9
            assert abs(run['eval'][-1]['return'] - float(row['latest_eval_mean_return'])) < 1e-5
    assert set(data['hp']) == set(data['loss'])
    return data


def rolling(values, n=21):
    values = np.asarray(values, dtype=float)
    sums = np.r_[0., np.cumsum(values)]
    i = np.arange(len(values))
    start = np.maximum(0, i - n + 1)
    return (sums[i + 1] - sums[start]) / (i - start + 1)


def array(rows, field):
    return np.array([d[field] for d in rows], dtype=float)


def decorate(ax, metric):
    ax.set_xlabel('环境步数（百万）')
    ax.set_xlim(0, 1.5)
    ax.set_xticks(np.arange(0, 1.51, .3))
    ax.grid(True, alpha=.18)
    if metric in ('win_rate', 'elimination_rate', 'timeout_rate'):
        ax.set_ylim(-3, 103)
        ax.set_ylabel('胜率（%）')
    elif metric == 'loss':
        ax.set_yscale('log')
        ax.set_ylabel('TD loss（对数刻度）')
    else:
        ax.set_ylabel('平均回合累计 reward')
    ax.spines[['top', 'right']].set_visible(False)


def legend(fig):
    handles = [Line2D([0], [0], color=COLORS[r], linestyle=STYLES[r], lw=2.3, label=r) for r in RULES]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.98, .99), ncol=2, frameon=False)


def save(fig, out, name, pdf):
    fig.savefig(out / f'{name}.png', dpi=190, facecolor='white')
    svg = out / f'{name}.svg'
    fig.savefig(svg, facecolor='white')
    svg.write_text(re.sub(r'[ \t]+\n', '\n', svg.read_text(encoding='utf-8')), encoding='utf-8')
    pdf.savefig(fig, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True, help='Exported HP/LOSS training data directory')
    parser.add_argument('--out', type=Path, required=True, help='Directory for generated figures and summaries')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 11,
                         'axes.titlesize': 13, 'axes.unicode_minus': False, 'svg.fonttype': 'path'})
    data = load(args.source)
    metrics = ('loss', 'win_rate', 'return')
    titles = ('训练 loss', '评估胜率', '评估 reward')
    summary = []
    for rule in RULES:
        for task, run in data[rule].items():
            first, final = run['eval'][0], run['eval'][-1]
            summary.append({'rule': rule, 'task_id': task, 'seed': 1,
                            'final_step': final['step'], 'initial_win_rate_pct': first['win_rate'],
                            'final_win_rate_pct': final['win_rate'],
                            'last5_eval_win_rate_pct': np.mean([r['win_rate'] for r in run['eval'][-5:]]),
                            'best_eval_win_rate_pct': max(r['win_rate'] for r in run['eval']),
                            'final_eval_return': final['return'],
                            'last5_eval_return': np.mean([r['return'] for r in run['eval'][-5:]]),
                            'final_elimination_win_rate_pct': final['elimination_rate'],
                            'final_timeout_win_rate_pct': final['timeout_rate'],
                            'last21_train_loss_mean': np.mean([r['loss'] for r in run['train'][-21:]])})
    with (args.out / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(summary)

    aggregates = []
    with PdfPages(args.out / 'training_comparison.pdf') as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.9))
        fig.suptitle('全部 32 个任务：hp 与 loss 训练结果对比', x=.04, ha='left', fontsize=19)
        for ax, metric, title in zip(axes, metrics, titles):
            for rule in RULES:
                key = 'train' if metric == 'loss' else 'eval'
                runs = list(data[rule].values())
                # Use only the common observed interval; no extrapolation.
                lo = max(r[key][0]['step'] for r in runs)
                hi = min(LIMIT, min(r[key][-1]['step'] for r in runs))
                grid = np.linspace(lo, hi, 601 if metric == 'loss' else 61)
                aligned = []
                for run in runs:
                    y = array(run[key], metric)
                    if metric == 'loss':
                        y = rolling(y)
                    aligned.append(np.interp(grid, array(run[key], 'step'), y))
                mean = np.mean(aligned, axis=0)
                ax.plot(grid / 1e6, mean, color=COLORS[rule], linestyle=STYLES[rule], lw=2.3)
                for x, y in zip(grid, mean):
                    aggregates.append({'rule': rule, 'metric': metric, 'step': x, 'task_mean': y, 'n_tasks': 32})
            ax.set_title(title + ' · 任务等权平均')
            decorate(ax, metric)
        legend(fig)
        fig.text(.04, .025, '横轴为每个任务的训练进度；按环境步数插值对齐。loss 为 21 点后向均值；评估每点 32 局，单个训练 seed。', color='#555555', fontsize=10)
        fig.tight_layout(rect=(.025, .09, .99, .86))
        save(fig, args.out, '01_overall_32_tasks', pdf)

        def draw_tasks(tasks, name):
            fig, axes = plt.subplots(len(tasks), 3, figsize=(16.8, 3.55 * len(tasks) + 1.2), squeeze=False)
            title = '657 / 647 / 403：逐任务训练对比' if len(tasks) > 1 else f'任务 {tasks[0]}：hp 与 loss 训练对比'
            fig.suptitle(title, x=.04, ha='left', fontsize=19)
            for i, task in enumerate(tasks):
                for ax, metric, label in zip(axes[i], metrics, titles):
                    for rule in RULES:
                        rows = data[rule][task]['train' if metric == 'loss' else 'eval']
                        x, y = array(rows, 'step') / 1e6, array(rows, metric)
                        if metric == 'loss':
                            ax.plot(x, y, color=COLORS[rule], alpha=.12, lw=.6)
                            y = rolling(y)
                        ax.plot(x, y, color=COLORS[rule], linestyle=STYLES[rule], lw=2, marker='o' if metric != 'loss' else None, markersize=2.3)
                    ax.set_title(f'{task} · {label}')
                    decorate(ax, metric)
            legend(fig)
            fig.text(.04, .018, 'loss：浅线为原始值，粗线为 21 点后向均值。胜率 / reward：greedy 评估原始点（每点 32 局）。hp、loss 为两种超时判定规则。', fontsize=10, color='#555555')
            fig.tight_layout(rect=(.025, .07 if len(tasks) > 1 else .11, .99, .925 if len(tasks) > 1 else .82))
            save(fig, args.out, name, pdf)

        draw_tasks(TASKS, '02_selected_tasks')
        for task in TASKS:
            draw_tasks((task,), f'task_{task}')

        fig, axes = plt.subplots(3, 2, figsize=(13.8, 11.8))
        fig.suptitle('训练回合：探索策略下的胜率与 reward', x=.055, ha='left', fontsize=19)
        for i, task in enumerate(TASKS):
            for rule in RULES:
                rows = data[rule][task]['episodes']
                x = array(rows, 't_env') / 1e6
                for j, field in enumerate(('is_win', 'return_value')):
                    y = array(rows, field) * (100 if j == 0 else 1)
                    axes[i, j].plot(x, rolling(y, 100), color=COLORS[rule], linestyle=STYLES[rule], lw=2)
            for j, ax in enumerate(axes[i]):
                ax.set_title(f'{task} · 训练' + ('胜率' if j == 0 else ' reward'))
                decorate(ax, 'win_rate' if j == 0 else 'return')
        legend(fig)
        fig.text(.055, .02, '每条曲线为最近 100 个训练回合的后向均值；包含 epsilon-greedy 探索。训练回合与 greedy 评估分开显示。', fontsize=10, color='#555555')
        fig.tight_layout(rect=(.035, .06, .99, .94))
        save(fig, args.out, '03_training_episode_curves', pdf)

        fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.1))
        fig.suptitle('胜率来源：歼灭胜与超时胜', x=.04, ha='left', fontsize=19)
        for ax, task in zip(axes, TASKS):
            for rule in RULES:
                rows = data[rule][task]['eval']
                x = array(rows, 'step') / 1e6
                ax.plot(x, array(rows, 'elimination_rate'), color=COLORS[rule], lw=2, linestyle='-')
                ax.plot(x, array(rows, 'timeout_rate'), color=COLORS[rule], lw=2, linestyle=':')
            decorate(ax, 'win_rate')
            ax.set_title(f'任务 {task}')
        handles = [Line2D([0], [0], color=COLORS[r], lw=2, linestyle=s, label=f'{r} · {label}') for r in RULES for s, label in [('-', '歼灭胜'), (':', '超时胜')]]
        fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.99, .9), ncol=4, frameon=False)
        fig.text(.04, .025, '总胜率 = 歼灭胜率 + 超时胜率。hp 与 loss 使用各自的超时判定规则；相同颜色表示同一组。', fontsize=10, color='#555555')
        fig.tight_layout(rect=(.025, .1, .99, .79))
        save(fig, args.out, '04_win_sources', pdf)

        tasks = sorted(data['hp'], key=lambda t: max(data[r][t]['eval'][-1]['win_rate'] for r in RULES), reverse=True)
        fig, axes = plt.subplots(1, 2, figsize=(14.5, 11))
        fig.suptitle('全部 32 个任务：最终评估结果', x=.065, ha='left', fontsize=19)
        y = np.arange(len(tasks))
        for j, metric in enumerate(('win_rate', 'return')):
            for k, rule in enumerate(RULES):
                axes[j].barh(y + (k - .5) * .36, [data[rule][t]['eval'][-1][metric] for t in tasks], height=.34, color=COLORS[rule], label=rule)
            axes[j].set_yticks(y, tasks)
            axes[j].invert_yaxis()
            axes[j].set_xlabel('最终评估胜率（%）' if j == 0 else '最终评估平均回合 reward')
            axes[j].grid(axis='x', alpha=.2)
            axes[j].spines[['top', 'right']].set_visible(False)
        legend(fig)
        fig.text(.065, .018, '最终 checkpoint 约为 150 万环境步；每任务每组评估 32 局，seed=1。', fontsize=10, color='#555555')
        fig.tight_layout(rect=(.035, .05, .99, .94))
        save(fig, args.out, '05_final_all_tasks', pdf)

    with (args.out / 'overall_curves.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregates[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(aggregates)
    print(json.dumps({'runs': 64, 'eval_rows': 3904, 'selected': [r for r in summary if r['task_id'] in TASKS], 'output': str(args.out)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
