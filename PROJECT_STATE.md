# PROJECT_STATE

## 2026-07-05 当前阶段交接

### 1. 当前已完成的核心逻辑与进展

当前仓库处在 Phase C 的“数据生成与准入检查已建立，from-scratch 训练脚手架已恢复但尚未正式执行”的阶段。主证据路线是受控 synthetic data -> admission checks -> from-scratch training -> capacity/reasoning measurement，不依赖预训练权重作为核心实验依据。

已完成的核心代码包括：

- `phase_c_data.py`
  - 实现随机序列 family 与分层 DAG family 的确定性数据生成。
  - 随机序列以 `seed + split + sample_id` 为样本定位键，目标 token 均匀采样，用于纯记忆容量测量。
  - DAG 样本使用严格分层有向无环图，保证唯一 `s -> t` 路径、固定出度、可解析 token 序列和可恢复图结构。
  - 权威 DAG 序列格式为：

    ```text
    [BOS] [GRAPH] edge pairs [QUERY] s t [ANSWER] path [EOS]
    ```

    `[GRAPH]` 后每两个 token 表示一条 `u -> v` 有向边；答案路径省略重复 source，只输出 source 之后到 target 的节点序列。

- `generate_phase_c_data.py`
  - 提供 `preview`、`validate`、`write` 三类 CLI。
  - `validate` 可流式跑完整准入检查；`write` 可写出 `.jsonl.gz` shards 和 manifest。

- `phase_c_model.py`
  - 定义 decoder-only Transformer 模型族。
  - 当前 120M preset 为 `12 layers, hidden=896, 14 heads`；此前统计约为 `115.7M` non-embedding parameters、`117.1M` total parameters。

- `phase_c_training.py`
  - 实现 answer-only collation、causal LM loss、随机序列容量指标、可恢复样本流、checkpoint save/load。
  - 随机容量指标以 bit 为单位，服务于后续 `M_train = N H_R - NLL_train,total` 口径。

- `train_phase_c_random.py`
  - 随机序列容量训练入口已经存在。
  - 该脚本当前应视为“已写好、待用户确认后执行”，不要在新会话中自动启动。

已验证的输出与进展：

- `phase_c_outputs/admission_100k_L4_d2/admission_report.json`
  - `all_checks_passed: true`
  - `records_checked: 260000`
  - family counts：`random=130000`、`dag=130000`
  - split counts：`train=200000`、`validation=20000`、`test=40000`
  - `validation_error_count=0`
  - duplicate identity/key/target/graph counts 全部为 `0`
  - `H_R_bits=[3840.0]`
  - `H_L_bits=[4.0]`
  - DAG graph statistics：`node_counts=[20]`、`edge_counts=[32]`、`path_count_required=1`
  - 词频平衡检查通过，random targets、dag path nodes、dag nonpath nodes 均低于 `max_abs_z=8.0` 阈值。

当前默认参数策略：

- `V=1024`
- `S=384`
- `q=4`
- `N_train=100k/250k/500k/1M`
- `N_val=10k`
- `N_test=20k`
- `L={2,4,6,8}`
- `d={2,4}`
- `W=d+2`
- `b_i=1`

### 2. 已遇到并解决的关键 Technical Obstacles

**Obstacle A: DAG 数据容易出现捷径或非唯一路径。**

解决方式：使用严格分层 DAG。所有边只能从第 `i` 层指向第 `i+1` 层；主路径节点连接唯一正确后继，同时加入局部干扰分支；非主路径节点禁止连回后续主路径节点，特别是禁止连到 `t`。准入检查会验证唯一 `s -> t` 路径、路径长度、出度和拓扑性质。

**Obstacle B: 随机序列如果没有样本 key，会测到条件分布熵而不是逐样本记忆容量。**

解决方式：随机序列加入固定长度 key，`KEY` 只用于定位样本，不编码目标内容。训练集容量测量只看训练样本上的 NLL 降低，未见随机序列只作为不可泛化性检查。

**Obstacle C: 大规模准入检查不能把全部记录放进内存。**

解决方式：准入检查走 streaming 设计，重复检测用 SQLite-backed tracking；数据可写为 `.jsonl.gz` shards。已验证的 `100k train + 10k val + 20k test`、双 family 合计 `260000` 记录可以完成准入检查。

**Obstacle D: DAG token 序列必须可逆解析，否则训练和诊断会脱节。**

解决方式：`phase_c_data.py` 同时提供生成、解析、重排边、验证逻辑；测试覆盖边重排后答案不变、解析字段一致、唯一路径仍成立。

**Obstacle E: 训练执行边界曾经容易被误读。**

解决方式：当前约定是“代码可以写、可以保留、可以做轻量检查，但不要擅自启动 GPU 或长时间训练”。`train_phase_c_random.py` 存在不代表已经跑出训练结果。

**Obstacle F: Windows 下 `conda run` 可能误解析短参数。**

解决方式：Phase C 验证命令优先直接调用目标环境的 `python.exe`。此前可靠命令形态是直接调用环境 Python，而不是通过 `conda run` 包一层。

**Obstacle G: admission success 容易被误表述为 training success。**

解决方式：所有结果必须标注阶段。当前确认的是“数据生成/准入验证通过”，不是“模型训练完成”，也不是“容量曲线已经得到”。

### 3. 下一步 TODO 的精确技术路径

下一步建议严格分成四个阶段，不要跳步。

**TODO 1: 冻结当前数据 contract 并补齐最小复现实验入口。**

1. 读取并确认 `phase_c_data.py`、`generate_phase_c_data.py`、`test_phase_c_data.py`。
2. 保持当前序列格式、metadata 字段和准入报告字段不变。
3. 用小规模命令做 sanity check：

   ```powershell
   python generate_phase_c_data.py validate --family both --V 1024 --S 384 --q 4 --L 4 --d 2 --W 4 --train-size 100 --validation-size 50 --test-size 50 --base-seed 20260615 --output-dir phase_c_outputs/admission_smoke_current
   ```

4. 如果只改文档，不需要跑；如果改生成器或校验器，必须跑 `test_phase_c_data.py`。

**TODO 2: 先跑随机序列容量实验的极小 CPU smoke test，不进入正式训练。**

目标不是得到论文结果，而是确认 `train_phase_c_random.py` 的数据流、collator、loss、checkpoint 和 metrics 没断。

建议路径：

1. 阅读 `train_phase_c_random.py` 的参数入口，确认是否支持小模型、小 batch、小 step、CPU。
2. 如果缺少 smoke 参数，先加一个显式 `--dry-run` 或 `--max-steps` 极小配置，不改变正式训练默认值。
3. smoke test 只允许使用极小数据量和 CPU，输出目录命名为 `phase_c_outputs/training_smoke_*`。
4. 只报告“训练脚本 smoke test 通过/失败”，不要把它解释为容量实验结果。

**TODO 3: 用户确认后再执行随机序列容量正式训练。**

正式训练的技术目标是测 `M_train = N H_R - NLL_train,total` 和 bits/parameter。

建议路径：

1. 选定模型规模：先从能在本机显存稳定运行的最小规模开始，再扩展到 120M preset。
2. 对每个 `P` 扫不同 `N_train`，覆盖欠容量、过渡区和饱和区。
3. 每个配置记录：
   - exact command
   - git/workspace state 或文件快照说明
   - seed
   - model preset
   - `V/S/q/N_train/N_val/N_test`
   - checkpoint path
   - train NLL total、train NLL per token、val random NLL per token
   - `M_train`
   - bits/parameter
4. 只有训练曲线稳定后，才拟合 `M_max(P)=alpha P+c`。

**TODO 4: 随机容量口径稳定后，再进入纯 DAG 推理实验。**

建议路径：

1. 只用纯 DAG 数据，不混入随机序列。
2. 先做对角线条件：训练深度和测试深度相同，避免把泛化 gap 混进 `lambda` 测量。
3. 固定 `d` 和 `P`，扫 `L`；再换 `d`、换 `P`。
4. 主指标：
   - test NLL in bits
   - Exact Match
   - stepwise accuracy
   - first-error position
   - path validity rate
   - `lambda = (NLL - H_L) / L`
5. 只有当新图测试明显高于随机 baseline，并通过重命名/边序扰动/删图/错图控制，才能说模型学到了可迁移路径推理。

**TODO 5: 最后再讨论统一容量 claim。**

只有当随机容量 `alpha P` 和 DAG 推理 `lambda` 都稳定后，才讨论二者是否进入同一 capacity account。当前不要提前写成已经证明。

### 4. 新 Agent 会话中必须警惕的边缘情况

- 不要把 `phase_c_outputs/admission_100k_L4_d2` 当成训练输出。它是准入验证输出。
- 不要自动运行 `train_phase_c_random.py`。训练需要用户明确批准，尤其不要擅自占用 GPU。
- 不要下载预训练权重。当前主路线是 from-scratch。
- 不要把 `H_L=L log2 d` 写成严格 NLL 下界。它是结构搜索难度锚点和经验坐标，模型 NLL 可能低于该值。
- 不要把验证集随机序列的 NLL 用来计算记忆容量。容量口径必须在训练集上计算。
- 不要把新图 DAG 的表现和训练图表现混在一起。主结果应使用 IID 新图测试；同结构新命名和边序重排只是诊断。
- 不要在未控制输入长度、图规模、上下文窗口占用时声称“性能下降只由深度导致”。
- 不要引入非线性 `f(L)`、混合随机/DAG、非均匀出度、CoT/递归干预等扩展，除非当前阶段明确推进到后续实验。
- Windows 终端读中文文档时可能显示乱码；优先用 UTF-8 读取，例如 PowerShell 中设置 `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8` 并用 `Get-Content -Encoding UTF8`。
- 该目录未必是干净 Git 仓库。不要依赖 `git checkout` 恢复文件；修改前先看文件内容，必要时用直接文件备份或最小 patch。
- 如果用户说“不要跑”，含义通常是不要自动执行长任务/训练；不是删除训练代码，也不是拒绝写代码。
- 所有最终汇报必须清楚写明：哪些是已完成，哪些只是代码就绪，哪些还未运行。

