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

## 3. 关键文件

| 文件 | 作用 |
|---|---|
| `phase_c_data.py` | 随机序列生成、gzip JSONL 读取、unit 文件路径选择 |
| `generate_phase_c_data.py` | 数据生成 CLI，包含 `random-units` 命令 |
| `generate_random_units.sh` | Linux/bash 下的一键数据生成脚本 |
| `phase_c_model.py` | GPT-style 小模型 preset |
| `phase_c_training.py` | dataset、collator、loss、capacity metric、checkpoint |
| `train_phase_c_random.py` | 训练入口，支持在线生成和文件数据集两种模式 |
| `test_phase_c_data.py` | 数据生成与 CLI 测试 |
| `test_phase_c_training.py` | 模型、训练、checkpoint、文件数据集测试 |

---

## 4. 数据格式与目录

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

## 5. 环境准备（可选）

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

## 6. 生成数据

### Windows PowerShell：生成 100k train + 20k test

```powershell
D:\anaconda\envs\minimind\python.exe generate_phase_c_data.py random-units `
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
D:\anaconda\envs\minimind\python.exe generate_phase_c_data.py random-units `
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

## 7. 本地 smoke test

### 7.1 单元测试

```powershell
D:\anaconda\envs\minimind\python.exe -m unittest test_phase_c_data.py test_phase_c_training.py
```

通过标准：

```text
Ran 20 tests
OK
```

### 7.2 极小数据 CPU smoke

生成极小数据：

```powershell
D:\anaconda\envs\minimind\python.exe generate_phase_c_data.py random-units `
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
D:\anaconda\envs\minimind\python.exe train_phase_c_random.py `
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

## 8. 正式训练

### 单卡训练

```bash
CUDA_VISIBLE_DEVICES=0 python train_phase_c_random.py \
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
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train_phase_c_random.py \
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
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train_phase_c_random.py \
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

## 9. 推荐扫描网格

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

## 10. 输出文件

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

## 11. 验收标准

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

## 12. 常见误区

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

## 13. 推送前检查

```powershell
D:\anaconda\envs\minimind\python.exe -m unittest test_phase_c_data.py test_phase_c_training.py
git status
```

不要直接 `git add .`，避免把临时数据、训练输出、旧图脚本误提交。

建议提交：

```powershell
git add README.md generate_phase_c_data.py generate_random_units.sh phase_c_data.py phase_c_model.py phase_c_training.py train_phase_c_random.py test_phase_c_data.py test_phase_c_training.py
git commit -m "refactor: add file-backed random capacity workflow"
git push -u origin codex/random-capacity-refactor
```

---

## 14. 下一步计划

当前阶段完成后，再进入：

1. 训练多模型、多 `N` 容量曲线；
2. 汇总 `final_metrics.json`；
3. 画 `NLL vs N`、`memory_bits vs N`、`bits_per_parameter vs N`；
4. 拟合容量密度 `alpha`；
5. 再开始 DAG 推理实验。
