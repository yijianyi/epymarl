# TABX QMIX：服务器运行说明

本分支默认 `timeout_rule=hp`；共同代码也支持 `loss`，两规则均有测试。模拟器固定使用仓库内官方子模块 `third_party/TABX`，任务 JSON 和模型/数据输出路径由运行参数指定。训练划分、逐局种子和 CPU 队列协议沿用原方案；旧快照和旧采集数据不混入这次迁移。

## 1. 安装和指定路径

从 EPyMARL 根目录初始化子模块，再使用 Python ≥3.10 的独立环境。`requirements-tabx.txt` 是 CPU 实测版本组合；训练服务器先安装对应 CUDA 的 PyTorch 2.6.0，再安装该文件。初版使用 JAX CPU 模拟、PyTorch CPU 或 GPU 策略/更新，不安装官方 pyproject 中整套 CUDA 依赖。

```bash
cd /your/server/epymarl
export EPYMARL_ROOT="$PWD"
git submodule update --init --recursive
git submodule status
python -m venv .venv-tabx
source .venv-tabx/bin/activate
# CPU 测试；GPU 服务器在这里改用对应 CUDA 的 wheel 源
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-tabx.txt
```

运行前自行设置这些变量，所有命令都复用它们：

```bash
export TASK_BANK="data/tasks/1106_tasks_merged_multiagent.json"
export TASK_SHA256="775cee683a650a726f056f05a20b1dc67d9b09104ceff9fb26ca0d11b4ac6887"
export RESULT_ROOT="$EPYMARL_ROOT/results/tabx-advanced-bc-v1"
export RULE=hp
```

任务库可以放在项目的 `data/tasks/` 后随整目录迁移，也可以通过绝对路径使用外部文件。相对任务路径始终按 EPyMARL 项目根目录解析，和当前 shell 目录无关；任务 JSON 不随 Git 提交。

官方仓库为 `https://github.com/ku-dmlab/TABX.git`，当前子模块 commit 为 `f218a29009d53059cccb0c44ce358ef2ccc3d249`。启动时强制验证 `src.tabx.__file__` 属于该子模块，忽略机器残留的 `TABX_ROOT` / `PYTHONPATH`；外部 `env_args.tabx_root` / `--tabx-root` 会被拒绝，这两个兼容参数最多只能用于断言同一个子模块路径。没有初始化时会提示执行 `git submodule update --init --recursive`。普通 EPyMARL 环境仍不需要安装 JAX 或 h5py。

| 输入 | 训练参数 | CPU CLI 参数 |
|---|---|---|
| 任务库 | `env_args.task_bank`，必填 | `--bank`，涉及模拟器的命令必填 |
| 官方模拟器 | 默认项目内 `third_party/TABX` | 同一子模块，无需传路径 |
| 任务文件校验 | `env_args.expected_bank_hash` | `--bank-sha256` |
| 模型、日志、评估和数据根目录 | `tabx_output_root`，必填 | `--root`，必填 |
| 任务 ID | `env_args.key`，内部统一为字符串 | `--task` / `--tasks` |
| 实验规则 | `env_args.timeout_rule`，默认 hp | `--rule`，从环境 YAML 读取，默认 hp |
| 训练尝试名称 | `tabx_run_id`，默认生成唯一 ID | `--run-id` 定位一次训练 |

TABX 的 `local_results_path` 自动设为本次 run 目录。训练及 CPU CLI 默认规则均来自 `src/config/envs/tabx.yaml`；可用 `env_args.timeout_rule=loss` / `--rule loss` 显式覆盖。模型和数据均位于：

```text
RESULT_ROOT/
  tabx_semantics.json               # 官方 commit、参数 JSON hash、布局与裁决定义
  hp/<task_id>/<train_seed>/<run_id>/
    manifest.json                   # 任务/任务库 hash、版本、种子协议、部署信息
    config.json                     # 运行配置
    checkpoints/<实际环境步数>/
      agent.th  mixer.th  opt.th
      config.json  manifest.json    # 完整配置及 agent/config SHA-256
    evaluation_requests/<step>.json # 原 request_id、实际 checkpoint_step
    evaluations/<request_id>.json   # 每局回报、终局、HP、吞吐和 CPU 信息
    evaluations/partial/<request_id>/<seed>.json # 单局原子发布，恢复时复用
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
  _eval_requests/                   # 稳定 request_id 索引和按 ID 的文件锁
  _eval_queue/<rule>/               # 无 batch_id 的单模型文件队列
  _batches/<rule>/<batch_id>/eval_queue/ # 批次文件队列
```

当前训练、评估和示范采集统一使用官方 `advanced` 启发式对手（默认 ε=0.1）。切换对手后使用新的输出目录；已有不同参数的目录会被拒绝。`tabx_semantics.json` 将官方仓库 URL、子模块 commit、实际使用的 `physics/parameters/default.json`、`heuristic_policy/parameters/advanced.json` 的 SHA-256、观测布局版本及终局定义绑定到输出根目录。这些字段也写入 run、checkpoint、CPU 评估和 HDF5；加载快照前校验 JSON，即使只修改参数 JSON 而没有修改 Python 源码，也不能悄悄改变模拟器。

快照和 tiers 内的模型引用相对于 `RESULT_ROOT`。整体迁移后使用新的 `--root` 和 `--bank`，项目内仍初始化同一子模块。运行清单中的原始绝对路径只用于溯源。权重、配置、源代码内容及 JAX/jaxlib/PyTorch 版本在加载前校验；CPU/CUDA wheel 的本地版本后缀可以不同。

## 2. 锁定任务划分并检查环境

两规则共用**同一份** pilot 清单。已有外部划分时，`--split-file` 接收 `{"train":["0",...],"validation":[...],"test":[...]}`。先建立“可信父标识 → 子任务”表；即使父任务不在库里，同父子任务仍合并，一个交叉后代会连接两个家族。优先使用完整 canonical/content hash；旧编号或任务名必须带明确的来源命名空间，来源不明的记录单列供核查。家族整体按约 80/10/10 分配，预留 23、651、657、647 的家族到训练侧。外部划分如拆散家族，报错列出任务 ID 并另写 `.conflicts.json`，保留原始划分和已有清单。

划分完成后，只用训练集历史超时率确定分层：零值单列，正值按含重复值的经验三分位点分成低、中、高，相同数值不拆开。先补齐非空超时率组的覆盖，再平衡各组试点名额，同时考虑九种布局、阵容规模和区域数。清单记录边界、训练/验证/测试/试点组计数及历史值来源；这些历史值只用于选场景，QMIX 训练后的实际超时率需重新测量。旧版 `tabx-pilot-1` 不会被覆盖，请使用新的输出文件名。

```bash
python "$EPYMARL_ROOT/src/tabx_pipeline.py" pilot \
  --bank "$TASK_BANK" --output "$RESULT_ROOT/pilot-v2.json" --seed 0

python "$EPYMARL_ROOT/src/tabx_pipeline.py" check-bank \
  --bank "$TASK_BANK" --bank-sha256 "$TASK_SHA256" \
  --output "$RESULT_ROOT/structure-check.json"

python "$EPYMARL_ROOT/src/tabx_pipeline.py" check-env \
  --bank "$TASK_BANK" --bank-sha256 "$TASK_SHA256" --rule "$RULE" \
  --tasks 23,647,657,651 --steps 12 --output "$RESULT_ROOT/native-equivalence.json"
```

`check-bank` 检查全部实际 tasks 的字段、有限值、范围、两队数量、原始敌我顺序和区域数组形状；输出失败任务 ID，失败时返回非零状态。当前 6,624 个任务全部通过，共 271 种 `(n_ally,n_enemy,n_zone)`，结构报告也包含按真实尺寸分组的任务列表，供调度器复用编译缓存。

模拟器使用真实尺寸；23 为 `(2,3,0)`，647 为 `(10,10,2)`，657 为 `(4,4,10)`，651 为 `(5,5,5)`。QMIX 对外始终为 obs `[10,378]`、state `[340]`、avail_actions `[10,8]`。新布局 `tabx-entity-slots-v2` 把友方写入 0–9 槽、敌方写入 10–19 槽，观测的其他单位段按观察者循环顺序逐实体映射，区域段从 obs[318] / state[280] 开始，缺失槽补零。不能把原生向量直接在尾部补零。真实单位死亡后 present 仍为 1，alive 变为 0；空槽和阵亡槽仅开放 IDLE=7。

`check-env` 先打印实际导入位置、官方 commit 和参数 JSON 校验值，再逐步比较两份独立官方状态：真实状态树、从固定输入反解的原生观测/世界状态、动作 mask、稠密奖励，同时验证同 seed 重放。冷 reset、第一次原生 step（含编译）和适配 step 用时单独记录。

实验裁决保留“动作后歼灭优先于时限”。纯超时时 hp 分支使用**原始血量比例严格比较**，完全相等判敌方胜，loss 分支始终判友方负。固定版本官方 TABX 会先四舍五入到 4 位小数；例如 0.50004 对 0.5 在 hp 实验中获胜，而官方舍入后判负。该差异进入实验记录并有真实终局边界测试。训练奖励为 raw_team_reward − raw_done_reward + 实验终奖，终局仅结算一次。

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
  env_args.expected_bank_hash="$TASK_SHA256" env_args.timeout_rule="$RULE" \
  tabx_output_root="$RESULT_ROOT" seed=1 \
  t_max=3000 batch_size=2 buffer_size=4 save_model_interval=1000
```

无 GPU 的测试机再加 `use_cuda=False`。正式训练去掉冒烟时的 `t_max`、`batch_size`、`buffer_size` 和保存间隔覆盖，默认使用 1500k（150 万）步、batch 32、CPU replay buffer 256 回合、25k 保存间隔、512 步上限。探索率在前 80k 步从 1.0 降到 0.05，贪心评估 epsilon 为 0；学习率 0.00005、gamma 0.99、梯度裁剪 10，目标网络每 200 次梯度更新同步一次，不标准化奖励和回报。

`qmix_tabx.yaml` 通过 `mixer_init_scale: 0.001`、`mixer_layer_norm: true`、`mixer_layer_norm_eps: 0.000001` 启用 mixer 设置。QMixer 对所有 Linear 权重使用 gain=0.001 的正交初始化、偏置置零；hyper_w_1、hyper_w_final、V 的隐藏层均为 Linear → LayerNorm(eps=1e-6) → ReLU。归一化仅处理状态分支，混合权重仍取绝对值以保留单调性。这些配置随 checkpoint 保存，重建 learner 时使用相同结构。

第一次 rollout 前保存真正的 step 0；周期保存使用完整局结束后的真实 `t_env`；训练正常结束时先发布最终快照，再原子写入 `training_complete.json`，封存所有 checkpoint 的步数及 agent/config/manifest hash。中断或失败的训练没有完成标记。`training_summary.json` 必须显示 `learner_updates>0`，并记录环境步数、episode 数、时间、主存和 GPU 分配峰值。异步模式完全跳过训练进程内的测试 rollout。训练日志中包含歼灭胜率、超时胜率、纯超时率、首杀、双方末帧 HP 和局长。

每次训练 reset 都按 `SHA256(JSON([任务内容hash, train_seed, "train", episode_index]))` 的前 32 位派生该局的环境种子，episode_index 从 0 开始；run_id 和规则均不参与。每局重新初始化 JAX key，前面 episode 的步数不会影响后续局的起始种子。同任务、同 train_seed 的 hp/loss 第十局因此可以配对，动作和轨迹仍允许不同。实际使用的序列写入 `training_episodes.jsonl`；显式评估 seed 不消耗训练 episode_index。

首轮按该推荐配置启动新批次，先查看 `train_metrics.csv` 首批更新的 `q_taken_mean`、`target_mean`、`td_error_abs` 和裁剪前梯度，再用后期贪心 CPU 评估的歼灭胜率及平均回报判断训练效果。1500k 预算、80k 探索期和 256 回合经验池均是待验证的推荐值；实际收益需要完整学习曲线确认。每次启动产生独立 run；可指定 `tabx_run_id` 方便调度，但不能复用已有 ID。

`checkpoint_path` 接收具体快照目录或含数字步数子目录的父目录。它会加载 agent/mixer/optimizer，登记为**新的训练尝试**，不发布虚假的随机 step 0。replay、目标网络历史、更新计数和随机状态并未完整恢复，训练 episode_index 也从 0 开始，不能作为精确续训。正式评估和采集应使用下面的不可变快照入口。

## 4. CPU 评估和分档

在独立进程启动 worker，可先用 `--once` 排空当前队列；不加则持续消费。worker 在导入框架之前强制清空 GPU 可见设备、设置 JAX CPU 和线程数。`--cpu-affinity` 可指定调度器分配的核心列表，多个 worker 会分割这些核心。

```bash
python "$EPYMARL_ROOT/src/tabx_pipeline.py" eval-worker \
  --root "$RESULT_ROOT" --bank "$TASK_BANK" \
  --rule "$RULE" --workers 2 --threads 1
```

文件队列用按 request_id 的文件锁及同文件系统的原子重命名互斥领取；新输出目录不生成 SQLite 数据库。批量模式把所有请求持久登记在各 run 的请求清单中，再按 run 轮流放入有界 ready 队列；旧参数 tabx_max_pending_evals 仅作兼容保留，不跳过 checkpoint。执行并发由 worker 数控制。失败记录 traceback，可用 --retry-failed 重试。崩溃请求在 --lease-seconds（默认 3600 秒）后可重领；每个成功 seed 单独原子保存，恢复只执行缺失 seed。最终逐局 JSON 完整发布后才发送完成通知。

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
  --root "$RESULT_ROOT" --bank "$TASK_BANK" \
  --tiers "$TIER_FILE" --rule "$RULE" --episodes-per-tier 20 --shard-episodes 10

python "$EPYMARL_ROOT/src/tabx_pipeline.py" verify \
  --root "$RESULT_ROOT" --task 651 --rule "$RULE" --tiers "$TIER_FILE" \
  --expected-episodes 20
```

新分片 schema 为 `tabx-hdf5-2`，固定数组维度和两个 offsets 不变；属性中明确写入真实阵容、区域数、native_shape、unit_slots、布局版本和官方版本。旧分片不会被新 reader 当作同一种语义加载。相同冻结权重下，把 `--episodes-per-tier` 改成 500 会补采缺少的 seed，已经发布的局不会重采。不同 tiers/权重生成独立数据版本。每个 tier 有单 writer 文件锁，每个分片先写同盘临时文件、校验，再原子发布；中断后重复 collect 可协调已发布分片和清单。

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
python -m pytest -q tests/test_qmix_tabx.py tests/test_tabx.py tests/test_tabx_official.py tests/test_tabx_scheduler.py
# 再执行官方 TABX 物理状态/观测/稠密奖励和同 seed 重放对照：
TABX_BANK="$TASK_BANK" \
  CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python -m pytest -q tests/test_tabx.py

# 一个任务、hp/loss 各 80 步真实短训练，检查十局种子配对、训练中 select、最终快照入选：
TABX_RUN_SMOKE=1 TABX_BANK="$TASK_BANK" \
  python -m pytest -q tests/test_tabx_revision_smoke.py
```

短训练测试还会让独立 CPU worker 真实加载官方子模块、排空初筛队列，并用每条规则的最终模型各采两局，写入/校验 HDF5 后逐字段读回 EpisodeBatch。它仅验证链路，不生成达标的 medium/high 档位。

新增批量测试还会用 23、651、647 三个真实任务验证并发训练、共享 CPU worker、全部 checkpoint 初筛、CSV、日志、恢复及不生成 SQLite 文件。常规测试另验证已完成 9 份请求空轮询 100 次零历史访问、两个进程竞争只执行一次，以及评估中断后的状态一致性和缺失 seed 补齐。该本地测试仅在测试内模拟 GPU 分配、实际使用 CPU 训练，不是多卡性能或显存验收；生产 batch-train 仍强制要求 CUDA。

## 7. 恢复 32 任务试点前重新测量

保持已冻结的 [pilot-v2.json](artifacts/tabx/pilot-v2.json) 和三套划分，不根据新模拟器回报重新挑场景。旧 `revision-validation.json` 是迁移前的历史验收，不能代表官方环境的收益或吞吐。

本次实际结果见 [official-migration-validation.json](artifacts/tabx/official-migration-validation.json)、[全库结构报告](artifacts/tabx/official-bank-check.json) 和 [四任务原生对照](artifacts/tabx/official-native-check.json)：57 项回归及独立两规则短训练通过，各 10 局、9 次更新；官方 CPU worker 真实加载和两规则 HDF5 回读通过。记录中的两局随机动作 CPU 校准仅用于本机诊断，不替代服务器上 32 个任务的重新测量。

```bash
python "$EPYMARL_ROOT/src/tabx_pipeline.py" check-env \
  --bank "$TASK_BANK" --bank-sha256 "$TASK_SHA256" --rule "$RULE" \
  --pilot "$RESULT_ROOT/pilot-v2.json" --steps 12 --benchmark-episodes 20 \
  --output "$RESULT_ROOT/pilot-native-calibration-$RULE.json"
```

该校准按真实尺寸排序以复用 JIT，记录合法随机动作下的 CPU 步/秒、冷编译开销、纯超时率和回报，作为新随机基准的测量；它不代表 QMIX 中高档质量。随后在服务器恢复训练、CPU 评估和结束后选档，用新返回值重新确认中高档差距。GPU/CPU 策略数值对照、满编峰值和正式学习曲线仍需在训练服务器测量，通过这些结果再决定正式放量；不会自动启动 32 任务或全库长训练。

## 8. 批量训练：每个分支一个终端，只写日志

新增的 batch-train 复用上面的单模型训练、不可变 checkpoint 和 CPU rollout。每个 task × seed 独立训练；GPU 名额按每卡分配，CPU 评估池由整个批次共享。默认规则从当前 checkout 的 src/config/envs/tabx.yaml 读取，也可显式传 --rule。公共模块不包含 HP/LOSS 分支专属逻辑。

两个分支同时运行时，使用两个独立 checkout/worktree，各开一个终端，激活相应训练环境。两边可以使用同一个 RESULT_ROOT（同一台机器的本地文件系统），必须遵守整机 GPU 和 CPU 配额。不要切换正在运行的 checkout 的分支或修改其源码。

HP 终端先运行三任务冒烟：

~~~bash
python src/tabx_pipeline.py batch-train \
  --bank "$TASK_BANK" --root "$RESULT_ROOT" --rule hp \
  --tasks 23,651,647 --seeds 1 --gpus 0,1 --jobs-per-gpu 1 \
  --eval-workers 2 --eval-threads 1 --batch-id hp-smoke-01 \
  --set t_max=3000 --set batch_size=2 --set buffer_size=4 \
  --set save_model_interval=1000 --set learner_log_interval=200
~~~

LOSS 终端：

~~~bash
python src/tabx_pipeline.py batch-train \
  --bank "$TASK_BANK" --root "$RESULT_ROOT" --rule loss \
  --tasks 23,651,647 --seeds 1 --gpus 2,3 --jobs-per-gpu 1 \
  --eval-workers 2 --eval-threads 1 --batch-id loss-smoke-01 \
  --set t_max=3000 --set batch_size=2 --set buffer_size=4 \
  --set save_model_interval=1000 --set learner_log_interval=200
~~~

--tasks、--pilot、--task-list 必须且只能提供一个，不会默认遍历全库。正式训练去掉冒烟用的 --set，使用 --pilot artifacts/tabx/pilot-v2.json 和新的 batch ID，继承 qmix_tabx.yaml。--task-list 支持 JSON ID 数组，或以空白、逗号分隔的文本 ID 列表。--seeds 1,2,3 会展开每个任务的三个训练作业。

`bash scripts/run_tabx_pilot32.sh` 会使用运行时指定的 TASK_BANK、RESULT_ROOT、GPUS，顺序运行 HP/LOSS 的 32 任务试点，再采最终模型的配对贪心样本。训练参数继承 qmix_tabx.yaml，默认 batch ID 为 pilot32-1500k-s1；可用 BATCH_ID、SEEDS、JOBS_PER_GPU、EVAL_WORKERS 等环境变量调整批次和资源。新版 mixer 增加 LayerNorm 参数，使用新的批次和 run；历史模型的加载和评估使用生成它的代码及配置。

--jobs-per-gpu 是每卡同时训练数，例如两卡各 12 个名额最多训练 24 个模型；--eval-workers 是整个批次的 CPU 进程数。--eval-cpu-affinity 0-7,16-23 可指定评估核心，worker 会分配这些核心；训练线程数沿用启动环境。增大并发前，应以正式 batch_size=32、buffer_size=256 和满编任务测量整卡显存、主机内存、总吞吐与评估积压。

--eval-prefetch 默认 4，限制每个 CPU worker 预留的待领取份数。例如 8 个 worker 时，pending/ 最多 32 份，另有最多 8 份运行中请求。其余请求保留在 run 的持久登记清单中，由轮转队列公平放行；训练进程不等待 CPU，checkpoint 一个也不漏。独立启动的 eval-worker 也支持该参数，并接管该批次的评估放行，后续 select 阶段沿用同样的上限。

GPU 编号使用当前分配范围内的外部编号或 GPU/MIG UUID，不能重复。已有 CUDA_VISIBLE_DEVICES / NVIDIA_VISIBLE_DEVICES 限制时，不能越过该范围。调度器先用当前 sys.executable 独立探测每张卡；每个训练子进程再设置 CUDA_VISIBLE_DEVICES=<外部编号>、JAX_PLATFORMS=cpu 和 tabx_require_cuda=True，验证一次真实 CUDA 分配。进程内 cuda:0 对应它分到的那张卡。GPU 不可用时明确失败，批量入口不会回退到 CPU。run manifest 的 deployment 记录外部 GPU、设备名及 PID。

批次输出：

~~~text
RESULT_ROOT/_batches/<rule>/<batch_id>/
  batch.json                         # 冻结任务/seed、超参数、代码和任务库 hash
  state.json                         # 默认每五分钟的整批快照，含 snapshot_time_utc
  runtime.json                       # 调度器、worker 和活跃训练 PID，实时更新
  jobs/<job_id>.json                 # 单 job 变化时原子更新，恢复优先使用
  jobs.jsonl                         # 实时追加的小事件记录
  batch.log                          # 与终端相同的批次进度、训练采样及评估结果
  batch_summary.csv                  # 默认每五分钟生成，每行含 snapshot_time_utc
  eval-workers.log                   # CPU worker 完整输出/错误
  eval_queue/
    pending/                         # 有界 ready 队列，尚未允许领取的请求不放这里
    running/                         # 原子领取，记录 owner / worker PID
    finished/                        # 新完成通知，消费后移出
    failed/                          # 新失败通知，消费后移出
    archive/<complete|failed>/       # 已消费通知，按 request_id 去重
  logs/<run_id>/train.log             # 启动参数、PID、完整训练输出
RESULT_ROOT/<rule>/<task>/<seed>/<run_id>/
  train.log                          # 指向该 attempt 的完整日志
  train_metrics.csv                  # 仅所属训练进程追加
  training_episodes.jsonl             # 每局一行，含实际 epsilon
  eval_summary.csv                   # 每个 request 一行，重试不重复
  evaluations/<request_id>.json       # 保留逐局原始评估结果和设备信息
  evaluations/partial/<request_id>/<seed>.json # 中断后只补缺少的 seed
  evaluation_requests/requests.jsonl # 每个 checkpoint 的持久登记清单
  evaluation_requests/registered/    # 已登记但尚未放行的请求
~~~

train_metrics.csv 的列为 t_env、learner_updates、loss、td_error_abs、grad_norm_before_clip、grad_norm_clip、q_taken_mean、target_mean、lr、bc_loss、bc_weight、total_loss。`loss` 始终表示纯 TD loss，`total_loss = loss + bc_weight × bc_loss` 用于反传；没有示范时 bc_loss 留空、bc_weight=0、total_loss=loss。混合阶段的 TD 公式与基线一致，但样本来自“示范＋在线”，采样分布不同。由 learner 复用同一组计算值，按 learner_log_interval 采样，默认每 2,000 环境步最多一行；首次更新也会采样。梯度范数是裁剪前的值，阈值单列。逐局 JSON 保留原始回报、局长、末帧血量、终局原因和种子，增加本局动作选择实际使用的探索率。

新 run 的 manifest 记录 training_metrics_version=tabx-train-metrics-v3-td-bc-total。拆分日志不改变反传总损失、梯度或参数更新。旧版 BC 混合阶段 CSV 的 loss 仍表示当时的总损失，读取时不重命名、不补成 TD loss，也不允许向旧表头追加新版指标。v2 的 q_taken_mean/target_mean 已按有效 mask 元素数计算；更早版本的 Q 均值曾缩小 10 倍，历史 CSV 均保留原值。

eval_summary.csv 记录 request_id、checkpoint_step、phase、action_selection、episodes、wins、win_rate_pct、mean_return、return_std、elimination_wins、timeout_wins、pure_timeouts。胜率是 0–100 的百分数，回报为原始团队奖励之和的均值，return_std 为样本标准差（单局记 0）。评估晚返回也始终使用快照 manifest 的实际训练步数。每个 worker 只用本次新结果和既有 CSV 更新所属 run，不重读历史评估 JSON；run 写锁避免并发写坏和重复计数。批次汇总只由持有批次锁的调度器写入。

终端与 batch.log 按模型指标采样/评估返回输出关键行，并默认每 30 秒刷新批次概况：

~~~text
[TRAIN] rule=hp task=651 seed=1 gpu=0 step=25048 updates=142 loss=3.821 bc_loss= bc_weight=0 total_loss=3.821 td_abs=1.247 grad_norm_preclip=18.4 grad_clip=10
[EVAL] rule=hp task=651 seed=1 checkpoint_step=25048 phase=screen episodes=32 wins=8 win_rate=25.0% mean_return=-0.42 return_std=0.316 elimination_wins=6 timeout_wins=2
[BATCH] rule=hp gpu_slots=0:4/4,1:4/4 train_running=8 train_done=6 train_pending=18 train_failed=0 eval_registered_waiting=120 eval_ready=16 eval_running=4 eval_done=31 eval_failed=0
[MODEL] task=23 seed=1 train_status=complete eval_status=interrupted status=evaluation_interrupted eval_done=1 eval_expected=3 eval_remaining=2
~~~

batch_summary.csv 包含 train_status、eval_status、status、eval_expected、eval_done、eval_pending、eval_running、eval_failed、eval_remaining，以及 run ID/attempt、GPU/PID、退出码、真实训练步数/更新数、耗时、峰值显存（字节）、最后评估的 checkpoint 步数/阶段/局数/胜局/百分比胜率/原始回报及完整日志路径。state.json、CSV 和终端均通过 derive_job_status 生成状态。完整 CSV 保留原始数值。本次没有 PNG、Matplotlib 依赖或重画图命令。

待训练 job 使用 deque 补位，GPU 占用数和批次状态计数按事件更新。dirty_jobs 只重算受影响模型；训练日志使用字节游标，只读取新增完整行，模型结束后释放读游标。没有新事件时只检查活跃子进程和有界评估队列，不遍历所有任务或历史请求。

每个 job 的小 JSON 和 jobs.jsonl 实时更新。整批 state.json 和 CSV 默认每 300 秒生成一次，可用 --snapshot-interval 调整；正常结束或可控中断强制写最终版。--status-interval 默认 30 秒，只打印内存计数，不触发全量 CSV。查看快照时注意 snapshot_time_utc；恢复时读取较新的 jobs/<job_id>.json 并核对持久请求和训练清单。旧 worker 仍存活或其他调度器持有接管锁时，拒绝接管不会改写原状态或 CSV。

每个新 checkpoint 完整发布后只登记一次请求，最终保存同一步快照不会再入队。训练完成时通过 completed_checkpoints 校验封存清单，逐 checkpoint 核对一次并仅补缺失请求；恢复时再核对一次。正常轮询只看 pending/running 和未消费通知，CSV 使用缓存状态；已完成请求不会在空轮询时重读或重复入队。

模型整体 complete 要求训练清单有效，且其中每个 checkpoint 的准确初筛请求 ID 都有完整、有效结果。额外复核或历史 attempt 的结果不能替代当前 attempt 的初筛。训练中 eval_expected 为空；封存后以清单数为准。评估池退出时，有缺项的模型显示 evaluation_interrupted 并记录剩余数，已经全部完成的模型保持 complete。批次返回非零并逐模型列出缺项，不会把训练完成统一改成整体完成。

--resume 仍逐份校验 checkpoint 身份和评估 seed，但按 run 核对 eval_summary.csv：先收集该 run 所有有效初筛、复核、随机基准和留出结果，再在同一把汇总锁内读取一次 CSV。表头、顺序、请求 ID 及按 CSV 规则序列化后的字段值一致时不写盘，保留修改时间；缺行、重复行或数值错误时原子重写一次。没有完整结果且 CSV 不存在时不创建空表。某个 run 中途检查失败不会用部分数据覆盖汇总。正常 CPU worker 完成一份新评估仍即时更新 CSV、照常打印评估结果。

Ctrl+C 会停止派发并回收本批自己的训练/评估进程组；重新执行同一命令并加 --resume。任务、seed、规则、超参数、任务库及源码内容必须匹配；仍需提供相同的 --set。GPU 列表、每卡并发数、评估 worker/线程数可调整。训练和初筛均完成的模型直接跳过；训练完成但评估缺项时复用原模型和逐 seed 结果，仅补缺项。最终 JSON 已写好但通知或 state.json 未更新时，恢复会校验并修正计数。缺少有效训练完成清单的 attempt 保留归档，新建 attempt 从头训练，其旧结果不计入新模型完成数。若旧子进程仍存活，恢复会报出 PID；同一批次的文件锁防止两个终端同时启动。

此命令只完成训练与 checkpoint 初筛。后续继续用 select 生成随机基准、复核和留出评估请求；它们的 phase 在 eval_summary.csv 中单列。批次结束后，若要消费该批的 select 请求，显式启动：

~~~bash
python src/tabx_pipeline.py eval-worker \
  --bank "$TASK_BANK" --root "$RESULT_ROOT" --rule "$RULE" \
  --batch-id "$BATCH_ID" --workers 2 --threads 1
~~~

省略 --batch-id 的独立 worker 只处理无批次的单模型请求。select 的随机基准、复核和留出阶段使用同一文件队列；原请求 ID 算法、完整结果复用和单模型调用方式保留。部署文件系统需支持同文件系统原子重命名及文件锁；新队列不使用 SQLite。

## 9. 调度规模与显卡容量分别验收

可用 CPU 上的 8,000 个逻辑作业验证调度器，固定 24 个活跃名额，不需要 8,000 个真实场景。现有任务库仍是 6,624 个不同场景；使用多个训练 seed 可以组成超过 8,000 个训练作业，正式训练 8,000 个不同场景则需先更新任务库。

~~~bash
python -m pytest -q tests/test_tabx_scaling.py
python tests/test_tabx_scaling.py --polls 100 \
  --output artifacts/tabx/scheduler-scale-validation.json
~~~

基准使用相同的 24 个真实 CPU 等待进程，分别测量 800/8,000 逻辑作业的空轮询中位数、P95、CPU、每分钟写盘量及单独的全量快照耗时。结果见 artifacts/tabx/scheduler-scale-validation.json；逻辑任务和快照负载为轻量模拟，不代表真实 QMIX 的显存容量。常规测试还检查单模型事件只更新该 job、重复通知不重复计数、有界预取与恢复、过时整批快照恢复及拒绝接管不改写汇总。

真实服务器仍需用正式 batch_size=32、buffer_size=256 和 10 对 10 任务逐步比较每卡 1、2、4 个模型，再按整卡显存、主机内存、总训练步数/小时和评估积压决定是否升至 12。终端 eval_registered_waiting 持续增长说明 CPU 评估跟不上；增加 GPU 并发不能消除这部分积压。

## 10. advanced 示范 → BC → online QMIX（可选）

原版 `qmix_tabx` 默认保持 `tabx_bc=False`。打开后，先对现有共享 RNN 的 Q 输出做合法动作掩码交叉熵预训练，再同步 target agent/mixer，从第一个在线回合起完全由模型选动作。老师只在采集和独立评估中控制我方，敌方始终沿用训练环境的 `advanced`。

先验证 647 的老师，再采集 403、657；每任务默认 128 局、512 步、`hp`。`collect-expert` 和原 `collect` 独立，后者的 medium/high 仍指已训练 checkpoint 档位。

```bash
export DEMO_ROOT="$EPYMARL_ROOT/results/tabx-expert-demos-vs-advanced-v3"
python src/tabx_pipeline.py collect-expert \
  --bank "$TASK_BANK" --bank-sha256 "$TASK_SHA256" \
  --root "$DEMO_ROOT" --rule hp --task 647 --teacher advanced
# 复用相同命令可续采；增加 --episodes 可扩充，不覆盖已有 HDF5。
# 只识别 10 位数字加 .h5 的正式分片；.shard-*.h5 等临时文件会被忽略。
# 确认 647 老师有效后：
for task in 403 657; do
  python src/tabx_pipeline.py collect-expert --bank "$TASK_BANK" \
    --root "$DEMO_ROOT" --rule hp --task "$task" --teacher advanced
done
```

查看 `DEMO_ROOT/hp/647/advanced/teacher_summary.json` 的 `wins`、`elimination_wins`、`timeout_wins`。逐局 HDF5 v3 保存真实执行 actions、ε=0 的 teacher_actions/teacher_mask、奖励、终局、present/alive；manifest 记录任务/库 hash、官方实现与参数、老师实际 ε 和参数 JSON hash。advanced 官方默认 ε=0.1，expert=0.01；可用 `--epsilon 0.05` 显式覆盖。切换敌方（例如旧 medium → 当前 advanced），或更换 ε、主 seed、步数上限、任务库时使用新的 `DEMO_ROOT`。旧 medium 示范会被校验拒绝，需重新采集。已经发布但未写入 manifest 的有效分片会在续采时校验并补登记，不重复采集；正式分片损坏会报错。老师零胜时仍完整保存轨迹并标为 `review_teacher_before_bc`，训练默认拒绝零胜示范；可先用 `--teacher expert` 单独收集，再设置 `tabx_bc_teacher=expert`。

先只做 BC 和独立评估，拿到 647 的两个数（不进行 online 训练）：

```bash
JAX_PLATFORMS=cpu python src/main.py --config=qmix_tabx --env-config=tabx with \
  env_args.key=647 env_args.task_bank="$TASK_BANK" env_args.timeout_rule=hp \
  tabx_output_root="$RESULT_ROOT" tabx_bc=True tabx_demo_root="$DEMO_ROOT" \
  seed=1 t_max=0
```

`bc_evaluations/after_bc_online_step_0.json` 在任何 online 动作/TD 更新之前生成。模型全程 greedy，老师保留采集时的 ε，两者使用同一组 128 个 `bc_holdout` seeds；与 `expert_collect` 采集种子分离，并显式检查无重叠。评估不增加在线环境步数、不推进训练 episode index，并恢复 PyTorch/NumPy 的探索 RNG。`checkpoints/0` 是 BC 后模型，manifest 标记 `policy_stage=after_bc_before_online`，不再是随机初始化模型。

第一轮 3 个任务 × 3 个训练 seed 的对照：

```bash
# 原版 QMIX
python src/tabx_pipeline.py batch-train --bank "$TASK_BANK" --root "$RESULT_ROOT" \
  --rule hp --tasks 647,403,657 --seeds 1,2,3 --gpus 0 \
  --eval-workers 2 --batch-id baseline-3tasks-3seeds
# BC + online QMIX
python src/tabx_pipeline.py batch-train --bank "$TASK_BANK" --root "$RESULT_ROOT" \
  --rule hp --tasks 647,403,657 --seeds 1,2,3 --gpus 0 \
  --eval-workers 2 --batch-id bc-3tasks-3seeds \
  --set tabx_bc=True --set tabx_demo_root="$DEMO_ROOT"
```

两组在线预算默认均为 150 万步。BC 总预算默认 1000 次更新、每批 32 个完整回合、Adam lr=0.0005、hidden_dim=128。先更新 400 次，再做三轮“模型 greedy 采集 32 局＋聚合池更新 200 次”，总预算仍为 1000 次。原始示范胜局目标抽样比例至少 0.5，失败轨迹保留；从第一轮 DAgger 起每批默认 16 局原始示范＋16 局 DAgger，DAgger 先均匀选轮次再在轮内均匀抽完整回合。BC 目标默认使用确定性老师标签，动作权重为 [1,1,1,1,3,3,3,1]。`bc_metrics.csv` 记录加权 loss、未加权 CE、accuracy，以及各动作 support/precision/recall；有效掩码同时排除死亡、空槽、终局帧、padding 和缺少标签的动作。

BC 模式在线 ε 从 0.1 线性降至 0.05，持续 10 万在线步。32 回合更新初始配比为 8 示范 + 24 在线，示范数按在线步数线性递减并取最近整数，模仿权重从 1.0 线性降至 0；10 万步后只从在线 replay 采样并使用原 TD 损失。示范池独立存于 HDF5，不占用 256 局在线 replay。BC 优化器独立，BC 更新数不计入在线 learner_updates。所有参数都在 `src/config/envs/tabx.yaml`，可用 `--set` 调整。调度器与训练器共用标准库配置函数 `tabx_config.configure_bc`；batch.json 的 training_config/identity 保存实际生效的 ε，overrides 保留用户原始覆写值。默认 BC 为 0.1 → 0.05 / 100000 步，关闭 BC 为 1.0 → 0.05 / 80000 步；自定义时使用 tabx_bc_epsilon_start、tabx_bc_epsilon_finish 和 tabx_demo_decay_steps。切换点和最终 greedy 复评分别在 `bc_evaluations/online_decay_complete.json`、`online_final.json`；10 万步按第一个跨过阈值的完整回合计算，保存实际步数，并发布该模型 checkpoint。

`bc_summary.json` 单列原始示范采集步数、DAgger 采集步数/回合数、各阶段 BC 更新预算、标签来源、动作权重与数据 manifest 校验和。每轮 DAgger 数据和采集模型位于 dagger/round_N，manifest 记录模型哈希、轮次、seeds、分片 SHA-256 和胜负；三轮共用同一个 BC 优化器，其最终状态保存为 bc_optimizer.th。online t_env 从 0 开始，不计入 DAgger 的额外环境交互。标准 CPU screen 评估继续按原协议异步执行，便于与基线配对比较。BC 的 0 步/10 万步/最终专项评估在训练进程中同步执行，其交互成本不计在线预算。已有 checkpoint 的纯评估需设置 `tabx_bc=False`；BC 模式要求新训练尝试，暂不支持中途恢复 replay/BC 状态。

等两组 screen 评估完成后导出对比表：

```bash
python src/tabx_pipeline.py learning-report --root "$RESULT_ROOT" --rule hp \
  --tasks 647,403,657 --seeds 1,2,3 --online-budget 1500000 \
  --output "$RESULT_ROOT/bc_comparison"
```

输出 CSV/JSON 每个 run 各一行，包含老师采集/留出胜率、BC 后胜率、首次训练/greedy 胜利在线步数、前/末 5 次 online screen 胜率均值、10 万步附近的实际评估步数与胜率、示范步数。未观测指标留空；多次尝试不自动择优合并。报告先读取完整的逐回合记录，以首次达到 online_budget 的回合终点作为唯一 cutoff_step，再同时过滤训练记录和 checkpoint 评估。例如预算 100000、终点 99900/100020/100150 时，只统计至 100020；100150 才首次赢则该预算下首次训练胜利留空。budget_reached=true 表示已达到预算，false 时 cutoff_step 是当前最近的回合终点；历史 run 缺少逐回合日志时 cutoff_step=online_budget、budget_reached=null（未知），评估仅统计 step≤budget。待返回的异步评估不补零。检查 cutoff_step、final_eval_step 和评估数量，确认两组预算、训练进度与评估已对齐。

验证：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_tabx_imitation.py tests/test_tabx_bc_revisions.py tests/test_tabx_dagger.py
TABX_BC_SMOKE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 \
  python -m pytest -q tests/test_tabx_imitation.py tests/test_tabx_bc_revisions.py tests/test_tabx_dagger.py -k "official_teacher or real_batch or official_relabel"
```

官方子模块当前的 dataclass 默认值在 Python 3.11+ 会触发 mutable-default 错误；上述真实模拟器验证使用 Python 3.10。短流程 smoke 只验证功能衔接，不代表 647 等正式任务的效果。

## 11. 干净标签、动作加权与 DAgger

新采集的专家 HDF5 使用 `tabx-hdf5-3`：`actions` 永远是实际送给环境的动作；`teacher_actions` 是同状态下 advanced 关闭探索的建议；`teacher_mask` 表示实际步、真实存活单位上的有效标签。行为策略仍可使用 ε=0.1，行为和标签从同一份旧老师记忆计算，只提交一次记忆更新。RNN 的上一动作和 QMIX TD 都继续读取实际 actions，标签不替换历史输入。老师建议攻击/右转/左转（编号 4/5/6）时权重为 3，其余为 1；加权 CE 除以有效样本的目标权重总和。

新建示范目录执行第 10 节 `collect-expert` 即可同时采集行为和干净标签。旧 v2 分片仍支持读取，显式选择 `tabx_bc_target=actions` 时才以执行动作作为监督目标，并在 BC 汇总记录来源。默认 teacher_actions 模式会拒绝缺少确定性标签的数据。将已有示范转为新的标注版本，例如：

```bash
# OLD_DEMO_ROOT 是旧示范根目录；新 DEMO_ROOT 应尚未包含该任务的数据。
python src/tabx_pipeline.py label-expert --bank "$TASK_BANK" --rule hp --task 647 \
  --source "$OLD_DEMO_ROOT/hp/647/advanced" \
  --output "$DEMO_ROOT/hp/647/advanced"
```

重标注不重新执行轨迹：从存储观测还原原生观测、逐步推进每个单位的老师记忆。原始动作、观测、奖励、终局保持不变，源文件不修改；新数据记录源 manifest/分片校验和、任务、物理参数及老师版本。输入与输出目录必须分开。旧 medium 对手数据不能直接转为当前 advanced 对手训练数据，仍需按当前环境重新采集。

三组效果对照保持相同总计 1000 次 BC 更新、lr=0.0005、hidden_dim=128、BC batch=32。在第 10 节 `batch-train` 命令基础上，分别使用以下设置和不同 batch-id；三组可以复用同一份带干净标签的原始示范：

| 对照组 | 追加设置 |
|---|---|
| 原版 BC | `--set tabx_dagger_rounds=0 --set tabx_bc_target=actions --set 'tabx_bc_action_weights=[1,1,1,1,1,1,1,1]'` |
| 干净标签＋动作加权 | `--set tabx_dagger_rounds=0` |
| 干净标签＋动作加权＋DAgger | 使用默认配置（三轮，每轮 32 局） |

三组均需 `--set tabx_bc=True --set tabx_demo_root="$DEMO_ROOT"`。仅比较预训练可用 `main.py ... tabx_bc=True ... t_max=0`；注意默认模式在 online step 0 之前仍会进行三轮 DAgger 的环境交互。这些交互单独记账，预算最多为 96×512 环境步。增加 `tabx_bc_steps` 时按初始 40%、剩余在轮次间平分来分配更新次数，保持总和精确等于配置值；关闭 DAgger 后全部更新用于原始示范。

最终 BC 的独立 greedy 评估还保存 `model_teacher_agreement`：在模型自己经历的留出状态上，与 ε=0 advanced 建议比较的动作一致率、各动作支持数/精确率/召回率及混淆矩阵。三组沿用同一套 `bc_holdout` seeds，DAgger 使用单独的 `dagger_collect` seeds，并检查与原始示范和留出评估无重叠。`learning-report` 输出 `bc_target`、`bc_updates`、`dagger_steps`、攻击与转向召回率以及动作一致率；DAgger 的额外交互成本应和独立胜率一起比较。

进入 online QMIX 后，抽到的示范来自聚合池：TD 仍用真实动作与奖励，辅助 BC 使用同一套目标选择和加权函数；`online_bc_metrics.jsonl` 记录未加权 CE、各动作精确率/召回率与支持数。已有 TD/BC/total loss 三列语义保持不变，其中 BC loss 现在按当前显式配置的动作权重计算。所有起始权重和采集量仍需正式 647 对照验证，短流程测试只验证实现。
