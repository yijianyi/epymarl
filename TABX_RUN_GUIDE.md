# TABX QMIX：服务器运行说明

本分支默认 `timeout_rule=hp`；共同代码也支持 `loss`，两规则均有测试。任务 JSON、AnyTABX checkout 和输出目录均由运行参数指定。仓库不包含任务数据、模型或生成的数据集，也不会自动启动正式训练。

## 1. 安装和指定路径

使用 Python ≥3.10 的独立环境。`requirements-tabx.txt` 是 CPU 实测版本组合；训练服务器先安装对应 CUDA 的 PyTorch 2.6.0，再安装该文件。初版使用 JAX CPU 模拟、PyTorch CPU 或 GPU 策略/更新，不安装 AnyTABX pyproject 中整套 CUDA 依赖。

```bash
python -m venv .venv-tabx
source .venv-tabx/bin/activate
# CPU 测试；GPU 服务器在这里改用对应 CUDA 的 wheel 源
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-tabx.txt
```

运行前自行设置这些变量，所有命令都复用它们：

```bash
export EPYMARL_ROOT="/your/server/epymarl"
export TABX_ROOT="/your/server/AnyTABX"
export TASK_BANK="/your/server/tasks/1106_tasks_merged_multiagent.json"
export RESULT_ROOT="/your/server/outputs/tabx"
export RULE=hp
```

AnyTABX 接口按 `70a97a1d4ad7ab108a2961df79058acdd676951d` 验证。可以传 `env_args.tabx_root` / `--tabx-root`，也可以用 `TABX_ROOT` 或 `PYTHONPATH`。用脚本的绝对路径启动，避免预先导入 EPyMARL 自己的 `src` 包。普通 EPyMARL 环境不需要安装 JAX、AnyTABX 或 h5py。

| 输入 | 训练参数 | CPU CLI 参数 |
|---|---|---|
| 任务库 | `env_args.task_bank`，必填 | `--bank`，涉及模拟器的命令必填 |
| AnyTABX 源码 | `env_args.tabx_root` 或环境变量 | `--tabx-root` 或环境变量 |
| 模型、日志、评估和数据根目录 | `tabx_output_root`，必填 | `--root`，必填 |
| 任务 ID | `env_args.key`，内部统一为字符串 | `--task` / `--tasks` |
| 实验规则 | `env_args.timeout_rule`，默认 hp | `--rule`，默认 hp |
| 训练尝试名称 | `tabx_run_id`，默认生成唯一 ID | `--run-id` 定位一次训练 |

TABX 的 `local_results_path` 自动设为本次 run 目录。模型和数据均位于：

```text
RESULT_ROOT/
  hp/<task_id>/<train_seed>/<run_id>/
    manifest.json                   # 任务/任务库 hash、版本、种子协议、部署信息
    config.json                     # 运行配置
    checkpoints/<实际环境步数>/
      agent.th  mixer.th  opt.th
      config.json  manifest.json    # 完整配置及 agent/config SHA-256
    evaluation_requests/<step>.json # queued 或 skipped_backpressure
    evaluations/<request_id>.json   # 每局回报、终局、HP、吞吐和 CPU 信息
    selection/<selection_id>/
      state.json                    # 可重入分档状态
      tiers.json                    # 达标后发布
      needs_training.json           # 不达标时的原因
    data/<tiers文件hash前缀>/<medium|high>/
      *.h5  manifest.json
    training_summary.json
    training_episodes.jsonl          # 每局 episode_index / episode_seed / 长度 / t_env
    training_complete.json          # 正常结束后原子发布的完整 checkpoint 清单
  _sacred/                          # Sacred 日志，仍在指定根目录内
  _jax_cache/                       # JAX 编译缓存，可用 JAX_COMPILATION_CACHE_DIR 覆盖
  eval_queue.sqlite3
```

快照和 tiers 内的模型引用相对于 `RESULT_ROOT`。移动整个输出目录后，使用新的 `--root`、`--bank`、`--tabx-root` 即可。运行清单中的原始绝对路径用于溯源，CPU worker 会使用命令行传入的新路径。权重、配置、源代码内容及 JAX/jaxlib/PyTorch 版本在加载前校验；CPU/CUDA wheel 的本地版本后缀可以不同。

## 2. 锁定任务划分并检查环境

两规则共用**同一份** pilot 清单。已有外部划分时，`--split-file` 接收 `{"train":["0",...],"validation":[...],"test":[...]}`。先建立“可信父标识 → 子任务”表；即使父任务不在库里，同父子任务仍合并，一个交叉后代会连接两个家族。优先使用完整 canonical/content hash；旧编号或任务名必须带明确的来源命名空间，来源不明的记录单列供核查。家族整体按约 80/10/10 分配，预留 23、651、657、647 的家族到训练侧。外部划分如拆散家族，报错列出任务 ID 并另写 `.conflicts.json`，保留原始划分和已有清单。

划分完成后，只用训练集历史超时率确定分层：零值单列，正值按含重复值的经验三分位点分成低、中、高，相同数值不拆开。先补齐非空超时率组的覆盖，再平衡各组试点名额，同时考虑九种布局、阵容规模和区域数。清单记录边界、训练/验证/测试/试点组计数及历史值来源；这些历史值只用于选场景，QMIX 训练后的实际超时率需重新测量。旧版 `tabx-pilot-1` 不会被覆盖，请使用新的输出文件名。

```bash
python "$EPYMARL_ROOT/src/tabx_pipeline.py" pilot \
  --bank "$TASK_BANK" --output "$RESULT_ROOT/pilot-v2.json" --seed 0

python "$EPYMARL_ROOT/src/tabx_pipeline.py" check-env \
  --bank "$TASK_BANK" --tabx-root "$TABX_ROOT" --rule "$RULE" \
  --tasks 23,651,657,647 --steps 8
```

`check-env` 输出真实导入位置、内容 hash、版本、10×378 局部观测、340 全局状态、10×8 动作 mask，并用同一 seed 重放。23 是 2 对 3、零区域；651 是 5 对 5；657 有 10 区域；647 是 10 对 10。

本批数据、seed=0 的新版清单见 [artifacts/tabx/pilot-v2.json](artifacts/tabx/pilot-v2.json)：训练 5,299、验证 663、测试 662；881 与 1275 同在训练集，1,984 个可信父标识均未跨集合，试点覆盖全部九种布局。

| 历史超时率组 | 训练集数量 | 试点数量 |
|---|---:|---:|
| 零值 | 2326 | 8 |
| 低：0 < rate ≤ 0.03125 | 1344 | 8 |
| 中：0.03125 < rate ≤ 0.0625 | 811 | 8 |
| 高：rate > 0.0625 | 818 | 8 |

提供的合并任务库保留了部分来源任务的旧 `metadata.canonical_hash`，例如 task 23 的该字段与当前 scenario/zone_scenario 的 SHA-256 不同。因此清单分别保存来源 canonical hash 和**实际任务内容 hash**，并记录两者是否相等；快照、种子、worker 和数据集使用实际内容 hash 校验身份。不会改写任务 JSON。可以额外设置 `env_args.expected_bank_hash` / `expected_task_hash` 来锁定输入；本批任务库的 SHA-256 为 `775cee683a650a726f056f05a20b1dc67d9b09104ceff9fb26ca0d11b4ac6887`。

## 3. 冒烟与正式训练

```bash
JAX_PLATFORMS=cpu python "$EPYMARL_ROOT/src/main.py" \
  --config=qmix_tabx --env-config=tabx with \
  'env_args.key="23"' env_args.task_bank="$TASK_BANK" \
  env_args.tabx_root="$TABX_ROOT" env_args.timeout_rule="$RULE" \
  tabx_output_root="$RESULT_ROOT" seed=1 \
  t_max=3000 batch_size=2 buffer_size=4 save_model_interval=1000
```

无 GPU 的测试机再加 `use_cuda=False`。正式训练去掉冒烟时的 `t_max`、`batch_size`、`buffer_size` 和保存间隔覆盖，默认恢复 500k 步、batch 32、buffer 512、25k 保存间隔、512 步上限。

第一次 rollout 前保存真正的 step 0；周期保存使用完整局结束后的真实 `t_env`；训练正常结束时先发布最终快照，再原子写入 `training_complete.json`，封存所有 checkpoint 的步数及 agent/config/manifest hash。中断或失败的训练没有完成标记。`training_summary.json` 必须显示 `learner_updates>0`，并记录环境步数、episode 数、时间、主存和 GPU 分配峰值。异步模式完全跳过训练进程内的测试 rollout。训练日志中包含歼灭胜率、超时胜率、纯超时率、首杀、双方末帧 HP 和局长。

每次训练 reset 都按 `SHA256(JSON([任务内容hash, train_seed, "train", episode_index]))` 的前 32 位派生该局的环境种子，episode_index 从 0 开始；run_id 和规则均不参与。每局重新初始化 JAX key，前面 episode 的步数不会影响后续局的起始种子。同任务、同 train_seed 的 hp/loss 第十局因此可以配对，动作和轨迹仍允许不同。实际使用的序列写入 `training_episodes.jsonl`；显式评估 seed 不消耗训练 episode_index。

正式初筛按任务 651、657、647，规则 hp/loss，`lr=0.00005/0.0002`，`epsilon_anneal_time=100000/300000` 的 24 种组合运行；先按服务器实测峰值安排训练 GPU 槽位。入围配置再做三个训练 seed 的复核。每次启动产生独立 run；可指定 `tabx_run_id` 方便调度，但不能复用已有 ID。

`checkpoint_path` 接收具体快照目录或含数字步数子目录的父目录。它会加载 agent/mixer/optimizer，登记为**新的训练尝试**，不发布虚假的随机 step 0。replay、目标网络历史、更新计数和随机状态并未完整恢复，训练 episode_index 也从 0 开始，不能作为精确续训。正式评估和采集应使用下面的不可变快照入口。

## 4. CPU 评估和分档

在独立进程启动 worker，可先用 `--once` 排空当前队列；不加则持续消费。worker 在导入框架之前强制清空 GPU 可见设备、设置 JAX CPU 和线程数。`--cpu-affinity` 可指定调度器分配的核心列表，多个 worker 会分割这些核心。

```bash
python "$EPYMARL_ROOT/src/tabx_pipeline.py" eval-worker \
  --root "$RESULT_ROOT" --bank "$TASK_BANK" --tabx-root "$TABX_ROOT" \
  --rule "$RULE" --workers 2 --threads 1
```

队列面向**同一台机器、本地磁盘**，用 SQLite 事务认领和每局唯一键防重。两个规则的 worker 并发数合计应落在整机 CPU 预算内。训练最多保留 `tabx_max_pending_evals=16` 个待处理请求，超额快照仍保存并记录跳过原因；选档会补齐缺失评估。失败记录 traceback；修复后以 `--retry-failed` 重启。进程崩溃的请求在 `--lease-seconds`（默认 3600 秒）后可重领；每局成功结果立即入库，重试只补缺失 seed。

从训练输出的 manifest 获取 `RUN_ID`，然后执行：

```bash
python "$EPYMARL_ROOT/src/tabx_pipeline.py" select \
  --root "$RESULT_ROOT" --task 651 --rule "$RULE" --run-id "$RUN_ID" \
  --baseline-episodes 64 --screen-episodes 32 --confirm-episodes 128 \
  --medium-range 0.4,0.6
```

训练期间，CPU worker 可以持续评估已发布的 checkpoint；此时 `select` 返回 `training_in_progress`，不创建选档状态、不冻结候选列表。只有检测到正常结束的 `training_complete.json`，才从该清单读取全部快照。训练到 25k 时调用过 select，再训练到 50k 结束，正式候选仍会包括 0、25k、50k。旧版只有 training_summary 的目录不会被自动认定为已完成。

正式 `select` 返回 `pending` 表示已复用原有评估、补交缺项，worker 完成后重复同一命令。状态目录和最终 tiers 均绑定训练完成清单的 SHA-256，读取时校验清单中所有快照。顺序为合法随机基准 → 所有快照共同种子初筛 → 相邻阶段支持的高档候选与中档候选独立复核 → 选定的一对权重再用保留种子确认。保留阶段也评估同 seed 随机基准，保存随机基准误差、实际 q、回报标准误和配对差值的正态近似 95% 区间。评估阶段按稳定 SHA-256 派生 seed，两个规则共用同阶段种子，采集阶段另行隔离。

达标时打印 `tiers.json` 的路径。没有稳定收益差距或缺中档时，写 `insufficient_quality_gap` / `missing_medium` 和补训信息；不会强行生成档位。可用 `--confirm-episodes 256` 提高精度；重做候选选择时增加 `--selection-round`，使用新的确认/保留种子，保留旧结果。正式两档需通过相同精度协议。

## 5. 采集、校验和报告

把 select 输出的实际路径设为 `TIER_FILE`：

```bash
python "$EPYMARL_ROOT/src/tabx_pipeline.py" collect \
  --root "$RESULT_ROOT" --bank "$TASK_BANK" --tabx-root "$TABX_ROOT" \
  --tiers "$TIER_FILE" --rule "$RULE" --episodes-per-tier 20 --shard-episodes 10

python "$EPYMARL_ROOT/src/tabx_pipeline.py" verify \
  --root "$RESULT_ROOT" --task 651 --rule "$RULE" --tiers "$TIER_FILE" \
  --expected-episodes 20
```

相同冻结权重下，把 `--episodes-per-tier` 改成 500 会补采缺少的 seed，已经发布的局不会重采。不同 tiers/权重生成独立数据版本。每个 tier 有单 writer 文件锁，每个分片先写同盘临时文件、校验，再原子发布；中断后重复 collect 可协调已发布分片和清单。

数据使用独立 `frame_offsets` / `step_offsets`：T 步对应 T+1 帧，满局恰好 513 帧；只保存实际执行动作，不导出 runner 为末帧补选的动作。每帧含 obs/state/avail_actions/present/alive；每步含 actions/reward/terminated；每局含 seed、长度、回报、胜方、终局原因、时限标记和末帧 HP。

`verify` 检查 SHA-256、形状/dtype、合法动作、末步终止位、回报重算、seed 数量/重复、真实末帧、轨迹重复率和压缩字节数，并将每个分片第一局通过 EpisodeBatch 回读逐字段比对。下游入口：

```python
from tabx_dataset import read_episode_batch
batch, lengths = read_episode_batch(shard_path, device="cpu")
# filled 包括末帧；实际动作范围为 [:lengths[i]]。
# batch["transition_mask"] 显式标记可训练的真实 transition。
```

这是本项目 HDF5 格式；MAICC 官方加载器还需要对应任务/shape 的适配。

```bash
python "$EPYMARL_ROOT/src/tabx_pipeline.py" report \
  --root "$RESULT_ROOT" --bank "$TASK_BANK" --pilot "$RESULT_ROOT/pilot-v2.json" \
  --output "$RESULT_ROOT/progress.json"
```

报告按规则列出有合格两档的任务、缺失/不达标任务、各 run 的实际更新数、中档 q、分档失败原因及采集计数。未达标任务不会计入 32×2×2×20=2,560 局或正式数据目标。

## 6. 测试

```bash
cd "$EPYMARL_ROOT"
python -m pytest -q tests/test_tabx.py
# 再执行真实 AnyTABX 物理状态/观测/稠密奖励和同 seed 重放对照：
TABX_BANK="$TASK_BANK" TABX_ROOT="$TABX_ROOT" \
  CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python -m pytest -q tests/test_tabx.py

# 一个任务、hp/loss 各 80 步真实短训练，检查十局种子配对、训练中 select、最终快照入选：
TABX_RUN_SMOKE=1 TABX_BANK="$TASK_BANK" TABX_ROOT="$TABX_ROOT" \
  python -m pytest -q tests/test_tabx_revision_smoke.py
```

测试覆盖 hp/loss 超时规则、末步歼灭、双灭、团队奖励仅替换一次、真实 EpisodeRunner 的 512 步终止位、短/满局和多局 offsets、HDF5 回读、队列超时重领与旧 owner 隔离、快照调度、防覆盖、换 run_id/规则/训练 seed 时的逐局种子行为、训练完成清单及评估复用、缺失父任务的兄弟家族与交叉家族、来源隔离、训练集三分位点和相同值不拆组。两项验收通过后，再在服务器安排 24 个调参作业；GPU/CPU 策略数值对照、满编峰值和正式学习曲线仍需在训练服务器测量，短冒烟不代表档位质量达标。
