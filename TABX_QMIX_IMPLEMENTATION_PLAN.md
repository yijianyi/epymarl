# EPyMARL × TABX 实施手册

目标是用已有的 6,624 个场景，从零训练 QMIX，在两条超时规则下**分别训练、分别选档、分别采集**。一个达标的任务最终有四组轨迹：血量分支的中档、高档，超时判负分支的中档、高档，每组 500 局。本手册中的新增文件名、CLI、配置和测试名均为待实现的接口约定。文中的命令要在相应步骤完成后运行。

建议将工作拆成五个可单独验证的提交：① 任务读取和环境术语/形状测试；② 裁决、环境注册及真实 runner 末步测试；③ QMIX 配置、快照和冒烟训练；④ CPU 评估队列、随机基准及分档；⑤ HDF5 写入、回读及 32 任务清单。五个提交都属于共同实现，两条实验分支对这五个提交使用同一版本。

## 1. 锁定起点、依赖和实验目录

- EPyMARL 工作目录为 `/home/xqyang/AIRS/epymarl`，核对时的提交为 `cbc38c0`。TABX 任务读取接口按 `/home/xqyang/AIRS/AnyTABX` 的 `70a97a1` 描述。将两个完整 commit、Python/JAX/jaxlib/PyTorch/h5py 版本写入运行清单。后者是能读取这批自定义任务的 TABX 代码版本。对照官方 TABX 的胜负定义，具体运行依赖固定到同一个版本。
- 任务库为 `/home/xqyang/AIRS/serverData/task_file/1106_tasks_merged_multiagent.json`，SHA-256 为 `775cee683a650a726f056f05a20b1dc67d9b09104ceff9fb26ca0d11b4ac6887`。让调度器遍历实际 `tasks`，全库 ID 为字符串 `"0"` 至 `"6623"`。两分支共用一份任务划分和种子协议。
- 使用满足本地 TABX 要求的 Python ≥3.10，在隔离环境中验证 EPyMARL、JAX CPU、PyTorch 和 h5py 可以共存。EPyMARL 的旧 requirements 和 TABX 的 CUDA 依赖要在实际训练机验证并锁版本。评估机必须能在无可见 GPU 时运行。先分别验证单任务 reset/step、PyTorch GRU 前向、HDF5 读写，再启动训练。
- 导入路径要实测：EPyMARL 的脚本用 `from envs ...` 等顶层导入，AnyTABX 内部用 `from src.tabx ...`，EPyMARL 自己也有名为 `src` 的包。建议用 `python /绝对路径/epymarl/src/main.py` 和相同方式启动新增 CLI，在进程启动时把 AnyTABX 仓库根目录放入 PYTHONPATH，避免再将 EPyMARL 仓库根目录作为 Python 包根加入。启动自检输出 `src.tabx.__file__` 和 `envs.__file__`，分别确认指向预期 checkout。将同一启动环境用于训练进程和新启动的 CPU worker。
- 输出根目录由配置传入，所有输出置于 `<root>/<timeout_rule>/<task_id>/<train_seed>/<run_id>/` 下。使用同一文件系统的临时路径发布 checkpoint 和分片，防止 worker 读到半成品。

**Git 顺序。** 先在一个开发分支实现共用代码并提交，记为 `COMMON_COMMIT`。从这个提交建立 `codex/tabx-timeout-hp`、`codex/tabx-timeout-loss`，分别提交环境 YAML 中的默认规则 `hp` 和 `loss`。规则函数、训练器、导出器的代码两边相同。使用单目录时顺序切换；需要同时跑时，两条分支各用一个 worktree、独立输出目录。作业启动时读取配置并将规则写入清单，worker 加载快照前校验规则与作业一致。

## 2. 文件修改清单

下列路径相对于 EPyMARL 根目录。建议依次实施，每一步都能独立验收。

| 文件 | 动作与职责 |
|---|---|
| `src/envs/tabx_tasks.py` | 新增。索引任务库、校验任务身份，调用 TABX 的任务构造器，返回一个任务的 env_params 和固定 shape。支持进程内缓存。 |
| `src/envs/tabx_outcome.py` | 新增。把本步更新后的存活情况、血量比例、是否到时限映射为终局结果和终局奖励。两条规则只在这里分岔。 |
| `src/envs/tabx_wrapper.py` | 新增。实现 EPyMARL MultiAgentEnv，管理 JAX key、状态、友方动作、敌方 heuristic、帧级 mask 及对局统计。 |
| `src/envs/__init__.py` | 增加 `REGISTRY["tabx"]`，在构造函数内部导入 TABX。接收 runner 已传入的 common_reward、reward_scalarisation。 |
| `src/config/envs/tabx.yaml`，`src/config/algs/qmix_tabx.yaml` | 新增。前者配置任务和规则，后者配置完整的 QMIX 参数。 |
| `src/run.py` | 提取 batch schema 构造逻辑供训练和评估共用；在 TABX 异步模式中跳过内联测试，保存初始及周期快照并发布评估请求。 |
| `src/runners/episode_runner.py` | 训练路径优先沿用现有实现。为独立评估/采集支持显式 episode seed 和可选逐帧记录；保持 `run()` 仍返回 EpisodeBatch。 |
| `src/tabx_pipeline.py` | 新增 CLI，提供 `check-env`、`eval-worker`、`select`、`collect`、`verify`。评估和采集共用同一条 rollout 路径。 |
| `src/tabx_dataset.py` | 新增。按 HDF5 规范写分片、校验分片，并读回一个可供下游消费的 EpisodeBatch。 |

首版选 `runner=episode` 和 `batch_size_run=1`；多任务训练通过并发运行多个作业实现。当确实启用 EPyMARL 的 ParallelRunner，再将 seed、终止语义和统计传递到其子进程并做等价测试。

## 3. 把一个固定任务接进 EPyMARL

**任务读取。** 建议接口为 `load_task(bank_path, task_id) -> task, manifest` 及 `build_env(task, manifest, seed, rule) -> wrapper`。按字符串 ID 查找，同时核对 canonical hash。每个长驻 worker 只解析一次 JSON；生产环境有多个进程时，可以预先生成不可变的小任务文件及索引，记录父文件哈希，避免每个短作业反复解析约 240 MiB。保留 `grid_info` 供溯源；环境输入复用任务中已经给出的单位、位置边界和区域数组。

调用 AnyTABX 的 `build_batched_env_params_from_tasks([task], physics="default", heuristic="expert", max_n_ally=10, max_n_enemy=10, max_n_zone=10)`。返回的参数含一个批维，对每个有批维的叶子取第 0 项，再交给单环境 reset。保留 `body_radiuss` 字段。空区域任务要验证空数组的形状可正确填充；若构造器要求二维输入，按 schema 在转换层整理空数组，不能伪造一个有效区域。

构造 `TABX(cfg, world_state_type="global", max_episode_steps=512, position_permutation=False)`，外包 `TABXEnemyHeuristicWrapper`。由 episode runner 显式 reset，不使用自动 reset，以便保存真正的末帧观测。在当前版本和槽位配置下，启动时断言：

| 对象 | 形状或取值 |
|---|---|
| `get_env_info()` | n_agents=10，n_actions=8，obs_shape=378，state_shape=340，episode_limit=512 |
| `get_obs()` | float32 [10,378]，按 ally_keys 顺序取局部观测 |
| `get_state()` | float32 [340]，取 observations["world_state"] |
| `get_avail_actions()` | 0/1 [10,8]，从当步环境状态读取。每行至少一个合法动作 |
| present / alive | 两个 [10] 掩码。present 排除占位单位；alive 标记本帧仍存活的真实单位 |

**reset。** 使用独立 episode seed 初始化 JAX key。每次调用 reset 或 step 分出新 key。显式 seed 的评估 episode 可单独重放；训练时不传 seed，则沿本作业可重放的 episode 序列前进。reset 返回 `(obs, info)` 并缓存原始 env_state、观测、世界状态、mask，t=0。

**step。** runner 传入的动作可能是 CUDA tensor；先转 CPU 整数数组，检查正好有 10 项。按 ally_keys 构造 JAX 动作字典，每次传一个新字典。空槽和阵亡槽在选择动作前仅开放 `IDLE=7`；其他槽采用环境的合法动作。训练中遇到非法动作应记录任务/步/动作并失败，避免导出的标签与实际执行动作不同。调用敌方 wrapper 后只取一次友方团队奖励，缓存下一帧的观测和 mask。

## 4. 明确一个规则裁决点，并处理第 512 步

建议第一版将实验裁决实现于 EPyMARL 的 TABX 适配层，固定外部模拟器源码。特别注意当前 AnyTABX `tabx.py` 的 step 在战斗更新之前取了 `parsed_state`，之后用其中的 `is_alives` 计算 `team_dones`，再刷新缓存。要按下列顺序**从返回的新状态**独立判定终局。用动作后的实际血量/存活状态和非 disabled 槽计算两队是否有存活单位，以动作后的 timestep 判断是否到 512。这样最后一步击杀可以在本帧结算。

| 优先级 | 条件 | is_win（友方） | end_reason |
|---|---|---|---|
| 1 | 两边都无存活单位 | false，两边都不给胜奖；沿用当前双灭的双方负奖 | mutual_elimination |
| 2 | 一方无存活单位，另一方存活 | 友方仍存活时为 true | elimination |
| 3 | 两边都存活，到时限，规则 hp | ally_hp_ratio > enemy_hp_ratio 时为 true，平局敌方胜 | timeout |
| 4 | 两边都存活，到时限，规则 loss | false | timeout |
| 5 | 其他 | 不结算胜方 | ongoing |

血量比例直接取新状态的 `game_manager.team_hp_ratio`。在该固定版本，原始友方 reward 等于稠密血量奖励加 `info["done_reward"]` 的友方终局分量。适配器可计算 `dense = raw_team_reward - raw_done_reward_team0`，再令 `team_reward = dense + outcome_terminal_reward_team0`。进行标量转换，按 team 0 取值；复用环境 win_reward / lose_reward。非终局终奖为零。适配器向 runner 输出自己的 terminated、胜负及统计，避免同时使用旧终局标记和新胜负。若以后改为直接修改 TABX.step，应移除适配层的替换逻辑，保证只结算一次。

**与现有 runner 对接的最小改法：** `step` 返回 `(next_obs, team_reward, outcome.ended, False, info)`。对所有已裁决终局令 `info["episode_limit"]=False`，单独设置 `hit_time_limit`。现有 runner 写入的 `terminated` 即为 1，Q learner 的末步 target 等于末步 reward。第 512 步歼灭时 `hit_time_limit=True` 且 `end_reason=elimination`。外部中断作业作废该残局，使用同一 seed 重试。runner 会对 info 的条目逐项相加，因此只传数值标量；结束原因在 info 里用编号或数值标志，导出清单再写字符串。数组、原始状态和 frame mask 通过 wrapper 的独立读取接口提供。

**本步验收。** 写四个可控末步输入：纯超时领先、持平、落后，以及最后一步恰好全歼敌方；两分支结果应分别为 胜/负、负/负、负/负、胜/胜。再测双灭、提前歼灭和未结束的普通步。验证团队奖励分解和末帧 training batch，特别检查 `terminated[T-1]==1`。同动作/同 seed 对照原生模拟器的物理状态、局部观测、稠密奖励；终局按本节实验规则对照预期。

## 5. 配置与训练入口

`main.py` 的合并顺序为 default → env → alg → Sacred 命令行覆盖。算法配置文件之间不会相互继承。复制 `src/config/algs/qmix.yaml` 为 `qmix_tabx.yaml`，然后设置下表键；保留原 QMIX 的 `action_selector=epsilon_greedy`、`learner=q_learner`、`agent_output_type=q`、`mixer=qmix` 等其余字段。名称用 `name: qmix_tabx`。

| 配置键 | 起始值 |
|---|---|
| runner / batch_size_run | episode / 1 |
| use_rnn / hidden_dim / obs_last_action / obs_agent_id | true / 128 / true / true |
| mixing_embed_dim / hypernet_layers / hypernet_embed / double_q | 64 / 2 / 128 / true |
| lr / batch_size / grad_norm_clip | 0.00005 / 32 / 10；另一候选 lr=0.0002 |
| epsilon_start / epsilon_finish / epsilon_anneal_time / evaluation_epsilon | 1.0 / 0.05 / 100000 / 0；另一候选退火 300000 步 |
| gamma / standardise_rewards / standardise_returns | 0.99 / false / false |
| buffer_size / buffer_cpu_only / target_update_interval_or_tau | 512 / true / 200 |
| save_model / save_model_interval / t_max | true / 25000 / 500000；入围实验延至 2000000 |

`src/config/envs/tabx.yaml` 建议内容（hp 分支示例）：

```yaml
env: tabx
env_args:
  key: "23"
  task_bank: /home/xqyang/AIRS/serverData/task_file/1106_tasks_merged_multiagent.json
  timeout_rule: hp
  max_episode_steps: 512
  world_state_type: global
common_reward: true
tabx_async_eval: true
```

让 `env_args.key` **直接代表 task_id**，wrapper 中统一 `str(key)`；当前 `main.py` 和 `run.py` 需要 map_name 或 key 来命名日志，已有 key 就可以正常进入。将具体规则和 task_id 写入 run 目录及配置。对老算法读取新增开关时用 `getattr(args, "tabx_async_eval", False)`。

**训练改动。**

1. 从 `run_sequential` 提取原有 `scheme, groups, preprocess` 构造，训练与 CPU rollout 共享；保留 `actions → actions_onehot` 预处理。训练用原 `ReplayBuffer + BasicMAC + QLearner`。注意 CPU 单独构建 MAC 时，传入含预处理派生字段的 `EpisodeBatch.scheme`，否则 `obs_last_action=True` 时缺 `actions_onehot`。
2. 在首次 `runner.run(False)` 前发布 step 0 的随机初始化快照；已有恢复作业不得覆盖 step 0。训练每跨过 25k 步，等本局结束后按**真实 t_env**发布一次，结束时补存最后一个快照。原代码第一次在完成一局后就保存，不能把这次保存命名为初始化。保留 opt/mixer 文件供原有权重加载使用，CPU 评估只读 agent 和配置。
3. 保存顺序为：同盘临时目录 → learner.save_models → 写完整 resolved config、task hash、timeout_rule、训练种子、t_env、learner 更新数、环境/代码版本和 agent SHA-256 → 原子发布目录 → 写评估请求。异步模式跳过现有 `runner.run(test_mode=True)` 内联循环；普通训练保持原行为。不能仅调大 test_interval，因为当前代码在首局之后仍会触发测试。
4. 记录 `t_env`、`learner.training_steps`、平均局长、回报、首杀、歼灭胜率、超时胜率、纯超时率。现有 target 网络硬同步单位为 learner 更新次数。若以后增加训练 rollout 并行度，按每轮真实更新次数校准；不要仅用总环境步数比较训练量。
5. 让实验恢复分两类处理：评估/采集由不可变快照精确重跑；训练若只恢复现有模型文件，应登记为新的训练尝试。要精确续训还需保存 replay、目标网络、更新计数、探索进度和随机状态，不能用只含 agent/mixer/opt 的目录宣称完全续训。

**调参预算。** 先用任务 23 做一局训练路径和 checkpoint 冒烟（先以短预算调通，再恢复正式超参数）；再用 651、657、647 做 3 任务 × 2 规则 × 2 学习率 × 2 退火长度，共 24 次 500k 环境步初筛。注意 replay 至少积累 32 个完整 episode 才训练；短冒烟若要验证反向传播，可临时将 batch_size 调为 2，确认 training_steps > 0。入围配置用 3 个种子复核，优先固定两分支共用的超参数。长局收尾差时先查终局和更新次数，再试 gamma=0.995。

**训练循环参考伪代码**（仅示 TABX 分支的调度位置）：

```text
runner.setup(...); learner = QLearner(...)
publish_snapshot_and_enqueue(t_env=0)           # 初始化权重已经构建
next_save = 25000
while runner.t_env < t_max:
    episode = runner.run(test_mode=False)
    replay.insert_episode_batch(episode)
    if replay.can_sample(batch_size):
        sample = replay.sample(batch_size)
        learner.train(trim_to_filled_and_move_to_device(sample), ...)
    if runner.t_env >= next_save:
        publish_snapshot_and_enqueue(t_env=runner.t_env)
        next_save = (runner.t_env // 25000 + 1) * 25000
publish_final_if_step_not_already_published()
```

实际代码从配置取 save_model_interval，用现有 buffer sample、trim 和 device 搬移逻辑；复用原日志与清理流程。原模型定时保存分支和 TABX 发布分支二选一，避免同一步重复写权重。

## 6. CPU 评估、冻结策略和两档选择

**进程模型。** 训练按 task × rule × train_seed 分配 GPU 作业槽；整台机器共享有界 CPU 进程池。worker 作为独立 CPU 进程启动，启动前设置 `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu`。将 args.device 固定为 cpu，配置 PyTorch 单进程线程数，给 JAX/系统库合理的 CPU 核心配额。对训练作业可显式选择 JAX CPU 模拟＋PyTorch GPU 策略/更新作为初版；训练环境吞吐需要时再比较其他部署。两条规则都记录同样的部署信息。测试 rollout 全程在 CPU；每卡训练作业数以 647 满编场景实测的编译和反向峰值确定。

**复用同一交互循环。** 给 `EpisodeRunner.run` 增加可选关键字 `episode_seed=None, recorder=None`，保持返回值不变；无显式 seed 的普通路径仍调用原来的 `env.reset()`。CPU 入口按快照配置构建 env、一个单局 EpisodeBatch 和 BasicMAC，加载 `mac.load_models(snapshot_dir)`，使用 `mac.agent.eval()`、`torch.inference_mode()` 和 `test_mode=True`。每局清空 GRU 隐状态，t=0 的上一动作输入为全零。首版每个 worker 同时跑一局；“32 局”表示该 checkpoint 最终累计 32 个 seed，不要求 32 个环境同时驻留。

每个评估请求包含 rule、task_id/hash、train_seed、run_id、checkpoint 完整路径/hash、seed 集合和用途。样本键至少包含 run/权重、评估阶段和 episode_seed；重试同一键采用相同种子。单机可用 SQLite 唯一键和事务认领，或每请求单一拥有者的文件队列；共享远程盘需验证锁/原子重命名语义。训练快于评估时限制待处理作业并降低下一轮常规评估频率；最终排名前先等候选评估完整，记录每个被跳过的快照。

**固定且相互隔离的种子。** 由 task hash、实验主种子、阶段、局序号作稳定哈希生成 seed，避免 Python 内置 hash。阶段分 train、随机基准、初筛、独立复核、正式采集；候选比较共用同阶段种子，两分支也共用种子表。若一次复核继续被用于挑选多个候选，则用另一组保留种子做最终确认。仓库里有多个训练种子/重试时，每个作业自己拥有独立训练随机流。

**选档算法。** 每行评估记录保存 `return=sum(raw_reward)`、长度、终局原因、is_win、末帧血量。不将胜率代入回报公式。

1. 每个 task、每条规则单独跑 64 局合法随机动作，得均值 R0。随机基准需在每个存活友方的合法动作集合内均匀取样。
2. 快照各用共同的 32 个种子贪心评估。将高回报且相邻训练阶段表现稳定的候选送入 128 局独立复核，得到高档权重及 RH。
3. 对待选权重求 q=(mean_return−R0)/(RH−R0)。中档以 0.5 为目标，首版接受 [0.4,0.6]。若没有命中，在学习曲线跨越该区间的位置加密 checkpoint，再训练或复跑该区间。绝对数值仅作为试点容差。
4. 对中、高候选用相同独立种子比较，报告回报均值、标准误及两档逐 seed 差值的区间。可将“配对差值的 95% 区间下界 > 0”作为试点验收；对 RH−R0 也要确认确有稳定提升，分母非正或接近评估噪声时不计算 q。必要时增至 256 局。保留随机基准误差和中档实际 q，不把归一化 1 当作理论最高回报。
5. 成功产出不可变的 `tiers.json`：两个 checkpoint/hash、q、R0、RH、均值/区间、种子协议、动作选择设置、任务和规则。失败记录 `insufficient_quality_gap` 或 `missing_medium`，进入补训清单。训练/评估任务数量和未达标数量均要报告。

## 7. 采集器、磁盘布局和回读

正式采集只接受与当前任务及规则匹配的 tiers.json，冻结权重、使用选档时相同的动作规则。评估与采集调用同一 rollout；后者增加 recorder，在 reset 后记录 frame 0，每步记录 transition 和返回的下一 frame。不要依据 filled.sum() 直接推断真实动作数。以 `runner.t` 或成功执行 step 的计数 T 为准；末帧 runner 补选的动作不进入导出结果。满 512 步的 episode 恰有 513 帧。

**建议每个分片内部按如下 HDF5 结构存储。** F 为分片所有 episode 的帧数总和，S 为实际步数总和，E 为局数。使用两个独立 offset 数组，避免第二局起错一帧。

| Dataset | dtype / shape | 含义 |
|---|---|---|
| frames/obs，frames/state | float32 [F,10,378]、[F,340] | 当前局部观测和全局状态 |
| frames/avail_actions | uint8 [F,10,8] | 当前帧的动作 mask |
| frames/present，frames/alive | uint8 [F,10] | 槽位存在、单位存活 |
| transitions/actions | int64 [S,10,1] | 真正送入环境的离散动作 |
| transitions/reward，transitions/terminated | float32 [S,1]，uint8 [S,1] | 友方标量奖励、真实训练终止位 |
| episodes/frame_offsets，episodes/step_offsets | int64 [E+1] | 起点含 0，末项分别等于 F 和 S |
| episodes 其余字段 | 每局一条 | seed、长度、原始回报、胜者、终局原因、是否达到时限、末帧双方 HP 比例 |
| 分片属性及清单 | schema_version 等 | task_id/hash、bank hash、rule、版本、档位、run_id、权重 SHA、分档指标和种子协议 |

每个分片单 writer，按小批 episode 分片、启用块压缩，实测压缩比。临时分片通过校验后再原子发布；作业崩溃仅重做未发布的 seed。用 `(task_hash,rule,tier,checkpoint_hash,episode_seed)` 唯一标识采集记录，禁止不同 worker 追加同一 HDF5 文件。32 任务试点的 20 局若与最终权重/行为协议完全相同，可计入正式 500 局并补采余量，否则生成新的数据版本。

**回读验证。** 根据 offsets 取出 T+1 帧、T 个动作。构造 EpisodeBatch 时先写有效 frame 字段，再写前 T 个 action、reward 和 terminated；按 EPyMARL schema 派生 actions_onehot 和 filled。另以 length 或显式 transition mask 限定有效动作。要验证短局、满 512 步、两局拼接三种情况。导入 MAICC 时需要 TABX 的任务索引和 shape 映射，HDF5 扩展名本身不保证被官方加载器直接读取。第一期验收至少做到从一份真实分片读回 batch 并逐项与原 episode 比对。

## 8. 建议实现的命令行及运行顺序

以下为开发完成后的 CLI 合约。两个 worktree 中分别运行，输出根可共用，但内部必须按规则/run_id 隔离。将路径和环境变量放在各自的启动脚本中，CPU worker 的变量只作用于该进程。

```bash
# 在当前分支的 EPyMARL 仓库根目录运行
export PYTHONPATH=/home/xqyang/AIRS/AnyTABX
TASK_BANK=/home/xqyang/AIRS/serverData/task_file/1106_tasks_merged_multiagent.json
RESULT_ROOT="$PWD/results/tabx"
RULE=hp   # loss worktree 使用 loss

# ① 验证读取、环境接口、动作以及终局；打印两个库 import 的真实路径
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" python src/tabx_pipeline.py check-env \
  --bank "$TASK_BANK" --rule "$RULE" --tasks 23,651,657,647

# ② 冒烟训练。task 键按字符串传递，目录中记录 resolved config
JAX_PLATFORMS=cpu python src/main.py --config=qmix_tabx --env-config=tabx with \
  'env_args.key="23"' env_args.timeout_rule="$RULE" seed=1 \
  env_args.task_bank="$TASK_BANK" local_results_path="$RESULT_ROOT" \
  t_max=3000 batch_size=2 save_model_interval=1000

# ③ 在另一进程启动共享 CPU 评估队列。部署两个 worktree 时，
# worker 使用与任务相符的版本，共享整机 CPU 并发预算
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python src/tabx_pipeline.py eval-worker \
  --root "$RESULT_ROOT" --bank "$TASK_BANK" --rule "$RULE" --workers 2
```

先确认 step 0、后续快照和至少一次 learner 更新，再启动正式调参。下面示例是 24 个初筛作业中的一个。调度器显式分配训练 GPU，限制同时在卡上的作业数。切换分支时不能让正在运行的进程依赖一个会被切换的源码 checkout；需要两边同时跑就使用两个 worktree。

```bash
JAX_PLATFORMS=cpu python src/main.py --config=qmix_tabx --env-config=tabx with \
  'env_args.key="651"' env_args.timeout_rule="$RULE" seed=1 \
  env_args.task_bank="$TASK_BANK" local_results_path="$RESULT_ROOT" \
  lr=0.00005 epsilon_anneal_time=100000 t_max=500000
```

训练一个 task 后从 manifest 取得准确 RUN_ID。选档命令先确保 64 局随机基准和常规 checkpoint 初筛结果齐全，再提交候选的独立复核请求；worker 完成后生成 tiers.json。可以将 select 实现为可重入状态机：缺评估时只提交缺项并返回 pending，再执行同一命令即可继续。所有 seed 派生规则和中高档入选原因写入 tiers.json。

```bash
# RUN_ID 与 TIER_FILE 取自该次训练和选档命令实际输出
python src/tabx_pipeline.py select --root "$RESULT_ROOT" --task 651 \
  --rule "$RULE" --run-id "$RUN_ID" --baseline-episodes 64 \
  --screen-episodes 32 --confirm-episodes 128 --medium-range 0.4,0.6

CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu python src/tabx_pipeline.py collect \
  --root "$RESULT_ROOT" --bank "$TASK_BANK" --tiers "$TIER_FILE" \
  --rule "$RULE" --episodes-per-tier 20

python src/tabx_pipeline.py verify --root "$RESULT_ROOT" \
  --task 651 --rule "$RULE" --tiers "$TIER_FILE"
```

## 9. 每个阶段的放行标准

| 阶段 | 必须看到的结果 |
|---|---|
| A. 环境 | 23 为 2 对 3、零区域；651 为 5 对 5；657 有 10 区域；647 为 10 对 10。shape、有效单位、动作 mask 均与原始场景一致。任意 seed 可重放。第 4 节四种末步用例及双灭测试通过。 |
| B. 冒烟 | 在 task 23 实际完成至少一次反向更新、保存和加载快照。CPU evaluator 重放同一 checkpoint/seed 得到相符的本机 CPU 结果；GPU/CPU 在同一观测比较动作和数值误差。worker 看不到 GPU，并验证评估时训练端没有再跑内联测试。 |
| C. QMIX 初筛 | 24 个作业都记录训练种子、步数、真实更新数、环境与框架版本、峰值 GPU/主存、CPU 评估吞吐。入围配置进行三种子复核。若高档不优于随机，先调训练，暂停该任务的数据放量。 |
| D. 32 任务试点 | 按已有外部 train/validation/test 划分；没有则先按可追踪亲缘关系及 hash 分组，再按组划分。父子与交叉后代尽量同侧，报告无法追溯的关系。挑 32 个训练任务，覆盖布局、阵容、零/满区域和不同历史超时率。两分支共用列表。32×2×2×20=2,560 局是全部达标时的目标，对未达标任务如实报告。 |
| E. 数据验收 | 每个动作合法且实际执行；回报可由 step 奖励重算；T≤512；仅终局末步 terminated=1；包含真实末帧；无 seed 重复；回读取出的值匹配。报告分档分布和重复轨迹比例，同时记录压缩后每局字节数与环境步/秒。 |
| F. 正式采集 | 每个达标 task 每条规则各采 500+500 局。用 manifest 检查数量、缺失、重试和版本，单独公布无法分档的 task。更多任务按试点的实际计算/存储成本逐批放量。 |

一套任务库同时用于训练与研究泛化时，应在首次调参前锁定分组/划分，让试点 task 全部处在训练侧。下游测试对局、评估 seeds 与 MAICC 离线数据/检索库隔离。

**规模核算。** 满编试点先测 JAX 编译、512 步长局反向传播和 CPU 评估三种峰值，分别制定训练卡作业槽和 CPU worker 数。512 局 CPU replay 单是 float32 观测预分配约 3.70 GiB/训练进程，还要计入其余字段和模型。全库若都达标为 13,248,000 局；按历史确认评估的平均局长 222.44 估计，仅未压缩 float32 局部观测约 44.8 TB。QMIX 打出来的局长须用试点实测替换。这里的容量为预算估算。

## 10. 实现时对照的源码与论文

- [EPyMARL 配置及训练入口](/home/xqyang/AIRS/epymarl/src/run.py)，[EpisodeRunner](/home/xqyang/AIRS/epymarl/src/runners/episode_runner.py)，[BasicMAC](/home/xqyang/AIRS/epymarl/src/controllers/basic_controller.py)，[QLearner](/home/xqyang/AIRS/epymarl/src/learners/q_learner.py)。
- [TABX 任务构造](/home/xqyang/AIRS/AnyTABX/src/tabx/sample_task.py)，[TABX 战斗 step](/home/xqyang/AIRS/AnyTABX/src/tabx/tabx.py)，[敌方策略封装](/home/xqyang/AIRS/AnyTABX/src/tabx/wrappers/wrappers.py)。
- [MAICC 官方仓库](https://github.com/LAMDA-RL/MAICC)，[论文附录 C.2](https://arxiv.org/html/2511.10030v1#A3.SS2)。论文的 SMAC 采样设计为每个配置两档各 500 条；本项目采用任务内随机基准到稳定强策略的相对档位。
- [TABX 官方仓库](https://github.com/ku-dmlab/TABX)，[TABX 论文 QMIX 参数](https://arxiv.org/html/2602.01665v4#A6.SS4)，[JAX CPU 平台选择](https://docs.jax.dev/en/latest/config_options.html#jax-platforms)。
