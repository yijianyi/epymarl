# 训练结果对比

数据来源：`/home/xqyang/AIRS/serverData/epymarl/`。本目录仅保存分析产物，未修改源数据。

比较 `hp`、`loss` 两种超时判定规则，各 32 个任务，均为训练 seed=1，训练约 150 万环境步。每任务有 61 个 greedy 评估 checkpoint，每点 32 个评估回合，共核验 64 次训练、3904 条评估记录；最终评估与 batch_summary.csv 一致。

## 图表

- `01_overall_32_tasks.png`：全 32 个任务的平均训练 loss、评估胜率、评估 reward。
- `02_selected_tasks.png`：657、647、403 的三行三列对比。
- `task_657.png`、`task_647.png`、`task_403.png`：每个任务的独立三指标图。
- `03_training_episode_curves.png`：三个任务训练回合的胜率和 reward，保留 epsilon-greedy 探索影响。
- `04_win_sources.png`：歼灭胜与超时胜的评估曲线。
- `05_final_all_tasks.png`：全部 32 个任务最终评估胜率和 reward。
- 每张图同时提供 SVG；`training_comparison.pdf` 合并全部 8 页。
- `summary.csv`：64 次训练的初始、最终、末 5 次评估均值、最高胜率和胜利来源等汇总。
- `overall_curves.csv`：总览图中实际绘制的对齐后数据。

## 统计口径

横轴为各任务自身的环境步数，不是多任务累计总步数。总览对全部 32 个任务等权平均，使用线性插值对齐环境步数，只在全部任务均有观测的公共区间绘制，不做外推。总览中评估步数采用 61 个等距网格点，因此曲线终点与各任务最终 checkpoint 的直接平均可能有轻微差异。

loss 为训练 TD loss，采用对数纵轴；浅色线为原始数据，粗线为最近 21 个日志点的后向均值。全任务 loss 先在各任务内平滑，再按步数对齐取算术均值。不同任务独立图的纵轴自动缩放。

主图胜率 = wins / episodes，使用原始计数重算，避免 batch.log 中一位小数百分比的舍入误差。reward 为评估的 mean_return，即每回合累计 reward，并非单步 reward。独立任务评估曲线不平滑；训练回合曲线采用最近 100 回合的后向均值，起始不足窗口时使用已有回合。

657、647、403 优先使用完整精度 CSV，其他 29 个任务从 batch.log 提取，因此其 loss 与 reward 具有原日志的输出精度。只有一个训练 seed；图表不表达多随机种子的置信区间。

## 主要结果

以下为各任务最终 checkpoint 的原始评估值。

| 范围 | hp 胜率 | loss 胜率 | hp reward | loss reward |
|---|---:|---:|---:|---:|
| 32 任务等权平均 | 13.5742% | 9.5703% | -1.1966 | -1.2607 |
| 657 | 96.875% | 28.125% | 1.6935 | 0.0608 |
| 647 | 0% | 0% | -1.4050 | -1.4050 |
| 403 | 43.75% | 93.75% | -0.1913 | 1.1881 |

657：hp 最终 31/32 获胜，包含 9 局歼灭胜和 22 局超时胜；loss 最终 9/32 获胜，全部为歼灭胜。两组最终歼灭胜率均为 28.125%。hp 在未训练 checkpoint 的胜率已为 68.75%，loss 为 25%，说明判定规则本身已影响初始基线，不能将全部差距归因于训练能力。

647：两组 train_metrics.csv 和 training_episodes.jsonl 逐字节一致；eval_summary.csv 除 request_id 外全部一致。loss 下降、reward 改善，但 61 次评估胜率始终为 0%。

403：末 5 次评估的平均胜率为 hp 53.75%、loss 89.375%，对应平均 reward 0.0411、1.1427；曲线显示 hp 后期回落，loss 后期表现更好。

## 重新生成

使用已安装 numpy、matplotlib 和中文字体的 Python 环境：

```bash
python analysis/training_comparison_20261111/plot_comparison.py \
  --source "$RESULT_SOURCE" --out "$ANALYSIS_ROOT"
```

从 EPyMARL 根目录运行；运行前将 `RESULT_SOURCE` 和 `ANALYSIS_ROOT` 设置为本机的数据及输出目录。`--source` 和 `--out` 均为必填参数。
