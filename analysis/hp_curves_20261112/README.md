# HP 训练曲线

数据源：`/home/xqyang/AIRS/serverData/epymarl/hp`，源文件未修改。

本次核验全部 32 个任务、22,372 条训练 loss 和 1,952 条评估记录。
全部任务训练 seed=1，每任务有 61 个评估 checkpoint，每点 32 局。
最终评估步数、胜率和 reward 均已与当前批次汇总文件核对。

## 文件

- `01_overall_32_tasks.png`：全部 32 个任务的平均 loss、胜率、reward。
- `overall_loss.png`、`overall_win_rate.png`、`overall_reward.png`：各指标独立图。
- `task_657.png`、`task_647.png`、`task_403.png`：各任务的三指标图。
- `02_selected_tasks.png`：三个任务的九宫格对照。
- `03_training_episode_curves.png`：三个任务的训练回合胜率及 reward。
- 所有图均附 SVG 矢量版本。
- `summary.csv`：逐任务最终评估、末 5 次评估均值等。
- `overall_curves.csv`：总览图实际绘制的数据。
- `plot_hp.py`：重新生成图表的脚本。

## 统计口径

横轴为各任务自身的环境步数。总览使用全部 32 个任务等权平均，
按环境步数线性插值对齐，仅绘制所有任务均有数据的公共区间，不做外推。
因此总览曲线终点可能与各任务最终 checkpoint 的直接平均有细微差别。

loss 使用对数纵轴，粗线为最近 21 个日志点的后向均值，独立任务图中浅线保留原始值。
总览先在各任务内平滑 loss，再对齐并取算术平均。

胜率由 wins / episodes 重算，reward 为 mean_return，即回合累计奖励的均值。
主图使用 greedy 评估原始点，不平滑。训练回合另图展示，使用最近 100 回合后向均值，
包含 epsilon-greedy 探索；窗口不足时用已有数据。

657、647、403 使用完整精度 CSV，并与批次日志交叉核对。
其余 29 个任务从 batch.log 提取，保留日志输出精度。
只有一个训练 seed，图表不表达跨随机种子的置信区间。

## 当前最终 checkpoint 结果

| 范围 | 胜率 | reward |
|---|---:|---:|
| 32 任务等权平均 | 12.5% | -1.2341 |
| 657 | 81.25% | 1.1264 |
| 647 | 0% | -1.4374 |
| 403 | 59.375% | 0.2398 |

647 的 61 次评估胜率始终为 0%；403 后期波动较大，末 5 次评估平均胜率为 36.25%。
这些数值基于本次指定目录中的当前记录。

## 重新生成

从项目根目录运行：

```bash
/home/xqyang/miniconda3/envs/python_robotics/bin/python analysis/hp_curves_20261112/plot_hp.py \
  --source /home/xqyang/AIRS/serverData/epymarl/hp \
  --out /home/xqyang/AIRS/epymarl/analysis/hp_curves_20261112
```
