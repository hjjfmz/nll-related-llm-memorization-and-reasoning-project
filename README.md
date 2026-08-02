# 随机序列记忆容量实验说明

本仓库当前聚焦 Phase C 的第一步：**纯随机序列记忆容量标定**。目标不是先做 DAG 推理实验，而是先复现 Morris 等人在《How much do language models memorize?》中的核心测量范式：对同一参数规模的 from-scratch GPT-style 模型，扫描多个训练集规模 `N`，观察模型从欠拟合到记忆饱和的曲线，并估计容量密度 `alpha`，再与论文中的约 `3.6 bits/parameter` 对齐。

---

## 1. 重构目的

原先代码可以训练随机序列，但数据是在训练时按 `sample_id` 在线生成的，文件夹里看不到实际数据集；模型 preset 也偏大，导致本阶段很难判断问题来自代码、数据规模、模型规模还是训练策略。

本次重构解决三个问题：

1. **模型规模对齐论文主实验范围**  
   使用较小的 GPT-style from-scratch 模型网格，例如 `L1_H32`、`L4_H128`、`L8_H256` 等，删除 `350m/700m/1b` 这类本阶段不可控的大模型，仅保留 `120m_legacy` 作为历史对照。

2. **数据显式落盘，保证可复现**  
   随机序列数据按 `1k` 样本为一个 unit 写入 `.jsonl.gz` 文件。`train-units=5` 表示读取前 5 个 unit，即 5k 样本；`train-units=1000` 表示读取前 1000 个 unit，即 1M 样本。

3. **为容量曲线服务**  
   同一个模型 `P` 要扫描多个训练集规模 `N`，从欠拟合到饱和，最终观察：
   - train NLL 是否随 `N` 和训练过程下降；
   - test NLL 是否保持接近随机 baseline；
   - `memory_bits` 是否随 `N` 上升后进入平台；
   - `bits_per_parameter` 的平台值是否接近论文的 `3.6 bits/parameter`。

---

## 2. 当前实验边界

当前只做：

- 纯随机序列；
- from-scratch 训练；
- NLL / memory_bits / bits_per_parameter 容量标定；
- train/test 无泄漏检查；
- 多 `N` 扫描。

当前不做：

- DAG 推理实验；
- 混合随机 + DAG 训练；
- `lambda` 深度惩罚；
- CoT 实验；
- 自动多 GPU 队列调度；
- 论文图表最终绘制。

---

## 3. 关键文件与新目录结构

| 文件/目录 | 作用 |
|---|---|
| `phase_c/cli.py` | 新主入口，提供 `random train/resume/extend/eval/gen-data/inspect` 子命令 |
| `phase_c/experiments/e03_random_capacity/config.py` | 实验 3 随机容量：运行配置加载、`resume/extend/eval` 参数构造与防呆 |
| `phase_c/experiments/e03_random_capacity/commands.py` | 实验 3 随机容量：子命令分发 |
| `phase_c/experiments/e03_random_capacity/train.py` | 实验 3 随机容量：训练主循环与 DDP 训练入口 |
| `phase_c/experiments/e04_dag_reasoning/` | 实验 4 纯 DAG 推理极限测量（骨架占位，待实现） |
| `phase_c/experiments/e05_unified_analysis/` | 实验 5 统一账本分析（骨架占位，待实现） |
| `phase_c/data/core.py` | 随机序列、DAG、admission、JSONL/GZIP 数据核心实现 |
| `phase_c/data/cli.py` | 数据 preview/validate/write/random-units 命令入口 |
| `phase_c/data/random_units.py` | 固定 1k unit 随机数据集落盘与 manifest 生成 |
| `phase_c/models/config.py` | `ModelConfig` 定义与参数合法性检查 |
| `phase_c/models/presets.py` | 论文对齐的小模型 preset 网格 |
| `phase_c/models/transformer.py` | Decoder-only Transformer、attention、MLP、block 实现 |
| `phase_c/models/counting.py` | total / embedding / non-embedding 参数量统计 |
| `phase_c/models/inspect.py` | 查看模型 preset、参数量、non-embedding 参数 |
| `phase_c/training/collation.py` | answer-only label 构造 |
| `phase_c/training/datasets.py` | 在线随机数据集与文件 unit 数据集 |
| `phase_c/training/stream.py` | 可恢复、可 DDP 分片的训练样本流 |
| `phase_c/training/losses.py` | causal LM loss |
| `phase_c/training/evaluation.py` | NLL、memory_bits、bits_per_parameter 评估 |
| `phase_c/training/checkpoint.py` | checkpoint 保存、加载、RNG 状态兼容恢复 |
| `phase_c/experiments/e03_random_capacity/reporting.py` | 实验 3 随机容量结果绘图脚本 |
| `generate_random_units.sh` | Linux/bash 下的一键数据生成脚本 |
| `phase_c/tests/test_phase_c_data.py` | 数据生成与 CLI 测试 |
| `phase_c/tests/test_phase_c_training.py` | 模型、训练、checkpoint、文件数据集测试 |
| `phase_c/tests/test_phase_c_random_cli.py` | 新主入口、resume/extend/eval 语义测试 |
| `phase_c/tests/test_package_structure.py` | 防止包内重新依赖旧根目录模块的结构测试 |

### 3.1 实验 ↔ 代码 映射（论文写作导航）

对应 `project.md` 中的实验编号：

| 实验 | 内容 | 代码位置 |
|---|---|---|
| 实验 0 | 冻结数据 contract | `phase_c/data/core.py`（`RandomConfig`/`DagConfig`、序列格式、metadata） |
| 实验 1 | 数据与测量管线准入 | `phase_c/data/core.py`（`run_admission_checks`、`validate_record`）+ `phase_c/data/cli.py`（`validate` 命令） |
| 实验 2 | 训练脚本 CPU smoke | `phase_c/training/` + `python -m phase_c.cli random train --max-steps 1 --device cpu` |
| 实验 3 | 纯随机序列容量标定（已完成） | `phase_c/experiments/e03_random_capacity/`（`train/config/commands/reporting`） |
| 实验 4 | 纯 DAG 推理极限测量（待实现） | `phase_c/experiments/e04_dag_reasoning/`（骨架占位） |
| 实验 5 | 统一账本分析（待实现） | `phase_c/experiments/e05_unified_analysis/`（骨架占位） |

---

## 4. 新主入口与运行模式

推荐后续优先使用：

```bash
python -m phase_c.cli random <command> ...
```

### 4.1 查看模型参数量

```bash
python -m phase_c.cli random inspect --model L4_H128 --V 1024
```

重点看：

```text
parameters.non_embedding
```

### 4.2 生成随机 unit 数据

```bash
python -m phase_c.cli random gen-data \
  --V 1024 \
  --S 32 \
  --q 4 \
  --unit-size 1000 \
  --train-units 1000 \
  --test-units 20 \
  --base-seed 20260715 \
  --output-dir phase_c_random_data/V1024_S32_q4_seed20260715
```

### 4.3 新实验

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m phase_c.cli random train \
  --model L4_H128 \
  --dataset-root phase_c_random_data/V1024_S32_q4_seed20260715 \
  --train-units 10 \
  --test-units 5 \
  --epochs 300 \
  --eval-size 0 \
  --output-dir phase_c_runs/random_L4_H128_units10_e300
```

### 4.4 中断后严格恢复

`resume` 只恢复原实验，不允许改变训练定义参数。适合断电、SSH 断开、手动停训后继续原计划。

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m phase_c.cli random resume \
  --run-dir phase_c_runs/random_L4_H128_units10_e300
```

### 4.5 没到平台时追加训练

`extend` 是显式追加训练，不再把“恢复中断”和“增加 epoch”混在一起。默认策略为 `constant-min`，即使用原实验的 `minimum_learning_rate` 固定小 LR 继续训练。

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m phase_c.cli random extend \
  --run-dir phase_c_runs/random_L4_H128_units10_e300 \
  --extra-epochs 100 \
  --lr-policy constant-min \
  --output-dir phase_c_runs/random_L4_H128_units10_e300_extend001
```

### 4.6 手动停训后补最终评估

如果训练被手动停掉，`final_metrics.json` 不会自动生成。使用 `eval` 从 checkpoint 补评估：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m phase_c.cli random eval \
  --run-dir phase_c_runs/random_L4_H128_units10_e300 \
  --eval-size 0 \
  --output-dir phase_c_runs/random_L4_H128_units10_e300_eval
```

---

## 5. 数据格式与目录

修改后的模型preset集：

```python
MODEL_PRESETS = {
    "debug": ModelConfig("debug", 2, 64, 1),
    "L1_H32": ModelConfig("L1_H32", 1, 32, 1),
    "L1_H64": ModelConfig("L1_H64", 1, 64, 1),
    "L1_H128": ModelConfig("L1_H128", 1, 128, 2),
    "L1_H256": ModelConfig("L1_H256", 1, 256, 4),
    "L2_H32": ModelConfig("L2_H32", 2, 32, 1),
    "L2_H64": ModelConfig("L2_H64", 2, 64, 1),
    "L2_H128": ModelConfig("L2_H128", 2, 128, 2),
    "L2_H256": ModelConfig("L2_H256", 2, 256, 4),
    "L4_H32": ModelConfig("L4_H32", 4, 32, 1),
    "L4_H64": ModelConfig("L4_H64", 4, 64, 1),
    "L4_H128": ModelConfig("L4_H128", 4, 128, 2),
    "L4_H256": ModelConfig("L4_H256", 4, 256, 4),
    "L8_H32": ModelConfig("L8_H32", 8, 32, 1),
    "L8_H64": ModelConfig("L8_H64", 8, 64, 1),
    "L8_H128": ModelConfig("L8_H128", 8, 128, 2),
    "L8_H256": ModelConfig("L8_H256", 8, 256, 4),
    "L16_H32": ModelConfig("L16_H32", 16, 32, 1),
    "L16_H64": ModelConfig("L16_H64", 16, 64, 1),
    "L16_H128": ModelConfig("L16_H128", 16, 128, 2),
    "L16_H256": ModelConfig("L16_H256", 16, 256, 4),
    "120m_legacy": ModelConfig("120m_legacy", 12, 896, 14),
}
```

默认随机序列设置：

```text
V = 1024
S = 32
q = 4
unit_size = 1000
base_seed = 20260715
```

每条样本的信息量为：

```text
H_R = S * log2(V) = 32 * 10 = 320 bits
```

默认数据目录：

```text
phase_c_random_data/
  V1024_S32_q4_seed20260715/
    dataset_manifest.json
    train/
      1.jsonl.gz
      2.jsonl.gz
      ...
    test/
      1.jsonl.gz
      2.jsonl.gz
      ...
```

读取规则：

```text
--train-units 1     -> train/1.jsonl.gz                         -> 1k
--train-units 5     -> train/1.jsonl.gz ... train/5.jsonl.gz     -> 5k
--train-units 100   -> train/1.jsonl.gz ... train/100.jsonl.gz   -> 100k
--train-units 1000  -> train/1.jsonl.gz ... train/1000.jsonl.gz  -> 1M
```

不同训练集规模是严格前缀关系：`1k` 是 `5k` 的子集，`5k` 是 `100k` 的子集，便于画同一数据分布下的容量曲线。

---

## 6. 环境准备（可选）

### Windows 本地

推荐使用已有 conda 环境：

```powershell
python -c "import torch; print(torch.__version__)"
```

### Linux 服务器

确认 Python 和 torch：

```bash
which python
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

多 GPU 训练应使用 `torchrun`，不要直接用 `python` 期望自动使用多卡。

---

## 7. 生成数据

### Windows PowerShell：生成 100k train + 20k test

```powershell
D:\anaconda\envs\minimind\python.exe -m phase_c.cli random gen-data `
  --V 1024 `
  --S 32 `
  --q 4 `
  --unit-size 1000 `
  --train-units 100 `
  --test-units 20 `
  --base-seed 20260715 `
  --output-dir phase_c_random_data\V1024_S32_q4_seed20260715
```

### Windows PowerShell：生成 1M train + 20k test

```powershell
D:\anaconda\envs\minimind\python.exe -m phase_c.cli random gen-data `
  --V 1024 `
  --S 32 `
  --q 4 `
  --unit-size 1000 `
  --train-units 1000 `
  --test-units 20 `
  --base-seed 20260715 `
  --output-dir phase_c_random_data\V1024_S32_q4_seed20260715
```

### Linux/bash：使用脚本生成

默认生成 `1M train + 20k test`：

```bash
bash generate_random_units.sh
```

生成 `100k train + 20k test`：

```bash
bash generate_random_units.sh 100 20
```

指定 Python：

```bash
PYTHON_BIN=/path/to/python bash generate_random_units.sh 1000 20
```

---

## 8. 本地 smoke test

### 7.1 单元测试

```powershell
D:\anaconda\envs\minimind\python.exe -m unittest phase_c.tests.test_phase_c_data phase_c.tests.test_phase_c_training phase_c.tests.test_phase_c_random_cli
```

通过标准：

```text
Ran 28 tests
OK
```

### 7.2 极小数据 CPU smoke

生成极小数据：

```powershell
D:\anaconda\envs\minimind\python.exe -m phase_c.cli random gen-data `
  --V 32 `
  --S 6 `
  --q 4 `
  --unit-size 10 `
  --train-units 2 `
  --test-units 2 `
  --base-seed 1234 `
  --output-dir phase_c_random_data\smoke_V32_S6_q4_seed1234
```

跑 1 step：

```powershell
D:\anaconda\envs\minimind\python.exe -m phase_c.cli random train `
  --model debug `
  --dataset-root phase_c_random_data\smoke_V32_S6_q4_seed1234 `
  --train-units 2 `
  --test-units 2 `
  --device cpu `
  --dtype float32 `
  --micro-batch-size 1 `
  --gradient-accumulation 1 `
  --max-steps 1 `
  --eval-size 0 `
  --eval-batch-size 1 `
  --output-dir phase_c_runs\smoke_debug_1step
```

## 9. 正式训练

### 单卡训练

```bash
CUDA_VISIBLE_DEVICES=0 python -m phase_c.cli random train \
  --model L4_H128 \
  --dataset-root phase_c_random_data/V1024_S32_q4_seed20260715 \
  --train-units 100 \
  --test-units 20 \
  --epochs 20 \
  --eval-size 0 \
  --output-dir phase_c_runs/random_L4_H128_units100
```

### 两卡 DDP 训练

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m phase_c.cli random train \
  --model L4_H128 \
  --dataset-root phase_c_random_data/V1024_S32_q4_seed20260715 \
  --train-units 100 \
  --test-units 20 \
  --epochs 20 \
  --eval-size 0 \
  --output-dir phase_c_runs/random_L4_H128_units100
```

### 1M 数据训练示例

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m phase_c.cli random train \
  --model L4_H128 \
  --dataset-root phase_c_random_data/V1024_S32_q4_seed20260715 \
  --train-units 1000 \
  --test-units 20 \
  --epochs 20 \
  --eval-size 0 \
  --output-dir phase_c_runs/random_L4_H128_units1000
```

`--eval-size 0` 表示最终完整评估整个 train/test split。正式算容量时建议使用 `0`；快速调试时可以临时改成 `10000`。

---

## 10. 推荐扫描网格

第一轮不建议一口气跑满所有组合。推荐从小规模确认趋势：

```text
models = L1_H64, L2_H128, L4_H128, L8_H256
train_units = 1, 2, 5, 10, 30, 100, 300, 1000
test_units = 20
```

每个模型固定参数量 `P`，扫描多个 `N`：

```text
N = train_units * 1000
```

目标是得到同一模型下 `NLL / memory_bits / bits_per_parameter` 随 `N` 增长的曲线。

---

## 11. 输出文件

每次训练会在 `--output-dir` 下写出：

```text
run_config.json
train_log.jsonl
checkpoint_latest.pt
final_metrics.json
```

重点看：

```text
final_metrics.json
```

核心字段：

```json
{
  "train": {
    "nll_bits_per_token": 0.0,
    "dataset_entropy_bits": 0.0,
    "memory_bits": 0.0,
    "bits_per_parameter": 0.0,
    "samples": 0
  },
  "test": {
    "nll_bits_per_token": 0.0,
    "samples": 0
  },
  "parameters": {
    "non_embedding": 0
  },
  "formal_capacity_evaluation": true
}
```

---

## 12. 验收标准

### 代码链路验收

必须满足：

- 单元测试通过；
- `random-units` 能生成 `train/` 和 `test/` 的 `.jsonl.gz` 文件；
- 训练脚本能通过 `--dataset-root` 读取文件数据；
- CPU smoke 能生成 `final_metrics.json`；
- `train.samples` 与 `train-units * unit_size` 一致；
- `test.samples` 与 `test-units * unit_size` 一致。

### 数据正确性验收

必须满足：

- `dataset_manifest.json` 存在；
- train/test 使用不同 split；
- 不同 `train_units` 之间是前缀关系；
- `test` 不与 `train` 共用同一批样本；
- `V/S/q/base_seed/unit_size` 在 manifest 中可追踪。

### 实验合理性验收

对于随机序列任务：

- train NLL 应能下降；
- test NLL 应接近随机 baseline；
- 若 `V=1024`，test baseline 约为 `10 bits/token`；
- 若 `V=32`，test baseline 约为 `5 bits/token`；
- test NLL 明显低于 baseline 时，应优先怀疑数据泄漏或评估 bug；
- 小数据 smoke 的 `bits_per_parameter` 不应拿来和 `3.6` 比较。

### 容量标定验收

要声称复现 Morris 风格容量，需要满足：

1. 对同一个模型 `P`，扫描多个 `N`；
2. 数据总熵覆盖从欠拟合到接近/超过模型容量的范围；
3. `memory_bits` 随 `N` 上升后出现平台；
4. 平台值除以 `non_embedding_parameters` 得到稳定的 `bits_per_parameter`；
5. 多个模型规模的容量平台近似线性随 `P` 增长；
6. 拟合得到的 `alpha` 与论文 `~3.6 bits/parameter` 在合理误差内。

---

## 13. 常见误区

### 误区 1：小 smoke 的 `bits_per_parameter` 很低，所以代码不对

不一定。若数据总熵太小，即使模型完美记住，`bits_per_parameter` 也不可能接近 `3.6`。

上限为：

```text
max_bits_per_parameter = dataset_entropy_bits / non_embedding_parameters
```

### 误区 2：test loss 不下降是坏事

不是。随机序列没有可泛化结构，test loss 应接近随机 baseline。  
如果 test loss 明显下降，反而需要排查泄漏。

### 误区 3：`train-units` 是训练 batch size

不是。`train-units` 是数据文件数量：

```text
train-units = 1000 -> 1000 个文件 -> 1M 样本
```

训练 batch 由下面两个参数控制：

```text
--micro-batch-size
--gradient-accumulation
```

### 误区 4：直接用 `python` 就能多 GPU

不能。多 GPU 要用：

```bash
torchrun --standalone --nproc_per_node=2
```

---

## 14. 推送前检查

```powershell
D:\anaconda\envs\minimind\python.exe -m unittest phase_c.tests.test_phase_c_data phase_c.tests.test_phase_c_training phase_c.tests.test_phase_c_random_cli
git status
```

不要直接 `git add .`，避免把临时数据、训练输出、旧图脚本误提交。

建议提交：

```powershell
git add README.md generate_random_units.sh phase_c
git commit -m "refactor: add file-backed random capacity workflow"
git push -u origin codex/random-capacity-refactor
```

---

## 15. 下一步计划

当前阶段完成后，再进入：

1. 训练多模型、多 `N` 容量曲线；
2. 汇总 `final_metrics.json`；
3. 画 `NLL vs N`、`memory_bits vs N`、`bits_per_parameter vs N`；
4. 拟合容量密度 `alpha`；
5. 再开始 DAG 推理实验。
