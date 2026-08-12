# Phase C 实验指南

本仓库包含两类从零训练的合成任务：

- E03 `random`：随机序列记忆容量标定。它用于测量模型可压缩的训练集随机信息量。
- E04 `dag`：同一 DAG 图分布上的双任务路径规划。它比较没有逐步轨迹监督的 `outcome` 与显式轨迹监督的 `trace`。

本 README 面向在远程服务器直接生成数据、训练、续训和评估的成员。所有命令从仓库根目录执行。

## 1. 环境与入口

确认当前 Python 可用 CUDA：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

主入口：

```bash
python -m phase_c.cli random <command> ...
python -m phase_c.cli dag <command> ...
```

单卡使用 `CUDA_VISIBLE_DEVICES=0 python ...`；双卡使用：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m phase_c.cli dag <command> ...
```

正常退出训练进程后，GPU 显存会由进程自动释放。不要让两个训练命令写入同一个 `--output-dir`。

## 2. E03 随机容量实验（简要）

E03 已完成主要代码重构。随机数据每个 unit 固定为 1000 个样本，`--train-units 10` 即固定读取 `train/1.jsonl.gz` 到 `train/10.jsonl.gz`，也就是 10k 样本。

生成数据：

```bash
python -m phase_c.cli random gen-data \
  --V 2048 --S 64 --unit-size 1000 \
  --train-units 1000 --test-units 20 \
  --base-seed 20260715 \
  --output-dir phase_c_random_data/V2048_S64_seed20260715
```

训练示例：

```bash
CUDA_VISIBLE_DEVICES=0 python -m phase_c.cli random train \
  --model L4_H128 \
  --dataset-root phase_c_random_data/V2048_S64_seed20260715 \
  --train-units 10 --test-units 20 \
  --max-steps 1000000 --eval-size 0 \
  --output-dir phase_c_runs/random_L4_H128_N10k
```

E03 重点看 `final_metrics.json` 的 train `memory_bits` 和 `bits_per_parameter`。随机 test 的 NLL 应接近随机基线，不应被当作泛化成功。

## 3. E04 双任务是什么

每个样本都是一张分层 DAG，输入 prompt 固定为：

```text
[BOS] [GRAPH] u1 v1 u2 v2 ... [QUERY] s t [ANSWER]
```

图、起点 `s`、终点 `t` 都已给出；正确路径不在输入中。设路径有 `L` 条边、`L+1` 个节点：

- `outcome`：只预测第一跳 `p1`。这是主任务，模型必须从图中做出下一步规划，没有真实路径 token 可供 teacher forcing。
- `trace`：预测中间节点 `p1 ... p(L-1)`，但不重复输出已知终点 `t`。这是显式轨迹监督对照。

同一组 outcome 和 trace 必须使用完全相同的 `V/L/d/W`、数据根目录、train/validation/test units、模型、种子、优化和训练预算。唯一变化是 `--task` 和输出目录。

`L=4` 表示从 `s` 到 `t` 有 4 条边：outcome 的目标长度为 1，trace 的目标长度为 3。`d` 是每个节点的出度；`W` 是每层节点数，默认 `d+2`。

## 4. E04 完整操作流程

### 4.1 生成固定 DAG 数据

先一次性生成共享 canonical 数据。三个 split 必须同时生成：

```bash
python -m phase_c.cli dag gen-data \
  --V 2048 --L 4 --d 2 --W 4 \
  --unit-size 1000 \
  --train-units 100 --validation-units 10 --test-units 20 \
  --base-seed 20260715 \
  --output-dir phase_c_dag_data/V2048_L4_d2_W4_seed20260715
```

目录应包含：

```text
phase_c_dag_data/V2048_L4_d2_W4_seed20260715/
  dataset_manifest.json
  train/1.jsonl.gz ... train/100.jsonl.gz
  validation/1.jsonl.gz ... validation/10.jsonl.gz
  test/1.jsonl.gz ... test/20.jsonl.gz
```

文件数据带 manifest 和 SHA256 校验。训练时如果文件被改动、配置不一致或 split 不匹配，会直接报错。`--train-units 10` 总是取前 10 个 train unit，因此 10k 是 100k 的严格前缀，便于比较不同训练规模。

服务器可用脚本生成：

```bash
# 参数顺序：train units、validation units、test units、output dir
PYTHON_BIN=python bash generate_dag_units.sh 100 10 20 \
  phase_c_dag_data/V2048_L4_d2_W4_seed20260715
```

脚本的 `V/L/D/UNIT_SIZE/BASE_SEED` 可通过环境变量覆盖；未设置 `W` 时使用默认 `d+2`。

### 4.2 训练 outcome 主任务

单卡：

```bash
CUDA_VISIBLE_DEVICES=0 python -m phase_c.cli dag train \
  --task outcome --model L4_H128 \
  --dataset-root phase_c_dag_data/V2048_L4_d2_W4_seed20260715 \
  --train-units 100 --validation-units 10 --test-units 20 \
  --max-steps 100000 \
  --micro-batch-size 8 --gradient-accumulation 128 \
  --eval-interval 1000 --monitor-eval-size 2000 \
  --eval-size 0 \
  --output-dir phase_c_runs/e04_outcome_L4_H128_L4_d2_N100k
```

双卡仅替换命令前缀，其他参数不变：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m phase_c.cli dag train \
  --task outcome --model L4_H128 \
  --dataset-root phase_c_dag_data/V2048_L4_d2_W4_seed20260715 \
  --train-units 100 --validation-units 10 --test-units 20 \
  --max-steps 100000 \
  --micro-batch-size 8 --gradient-accumulation 128 \
  --eval-interval 1000 --monitor-eval-size 2000 \
  --eval-size 0 \
  --output-dir phase_c_runs/e04_outcome_L4_H128_L4_d2_N100k
```

### 4.3 训练 trace 对照任务

复制 outcome 命令，只改 `--task trace` 和输出目录。其余训练定义必须相同：

```bash
CUDA_VISIBLE_DEVICES=0 python -m phase_c.cli dag train \
  --task trace --model L4_H128 \
  --dataset-root phase_c_dag_data/V2048_L4_d2_W4_seed20260715 \
  --train-units 100 --validation-units 10 --test-units 20 \
  --max-steps 100000 \
  --micro-batch-size 8 --gradient-accumulation 128 \
  --eval-interval 1000 --monitor-eval-size 2000 \
  --eval-size 0 \
  --output-dir phase_c_runs/e04_trace_L4_H128_L4_d2_N100k
```

### 4.4 运行中看什么

训练输出写入 `--output-dir`：

```text
run_config.json          # 完整参数和任务类型
train_log.jsonl          # 训练 loss、学习率、梯度范数
eval_log.jsonl           # 周期性 validation 指标
checkpoint_latest.pt     # 最新 checkpoint
final_metrics.json       # 训练正常结束后的 train/validation/test 结果
```

周期监控只读 validation，绝不使用 test 来挑 checkpoint 或判断是否继续训练。

- outcome：在 `eval_log.jsonl` 看 `first_hop_accuracy`。随机选择基线是 `1/d`，例如 `d=2` 时为 0.5。
- trace：看 `free_run_trace_em` 和 `path_validity_rate`。它们是真实自回归生成指标，不会把真实路径 token 喂回模型。
- 两个任务都可看 `teacher_forced_nll_bits_per_token`，但不要把它当作 free-run 成功率。

`final_metrics.json` 只在训练正常结束时写入。若手动中断，先确认已有 `checkpoint_latest.pt`，再运行 `dag eval` 补全最终指标。

### 4.5 中断、恢复与加长训练

`resume`：原计划被中断后严格恢复。它读取 run config 和 checkpoint，不能改 task、数据、模型和训练定义参数。

```bash
CUDA_VISIBLE_DEVICES=0 python -m phase_c.cli dag resume \
  --run-dir phase_c_runs/e04_outcome_L4_H128_L4_d2_N100k
```

`extend`：原训练已经结束或明确想继续更久时使用。它以保存的 `minimum_learning_rate` 固定小学习率追加训练，避免重新按总步数计算 scheduler 导致学习率跳变。

```bash
CUDA_VISIBLE_DEVICES=0 python -m phase_c.cli dag extend \
  --run-dir phase_c_runs/e04_outcome_L4_H128_L4_d2_N100k \
  --extra-steps 25000 \
  --lr-policy constant-min \
  --output-dir phase_c_runs/e04_outcome_L4_H128_L4_d2_N100k_extend001
```

说明：E04 双任务重构前的旧 DAG run 没有保存 `task`，不能用新命令续训；请重新生成正式数据并重新开始 E04 run。

### 4.6 最终评估与边序对照

对完整 test split 做最终评估，并测试模型是否依赖边在 prompt 中的排列顺序：

```bash
CUDA_VISIBLE_DEVICES=0 python -m phase_c.cli dag eval \
  --run-dir phase_c_runs/e04_outcome_L4_H128_L4_d2_N100k \
  --eval-size 0 --edge-reorder-seed 123 \
  --output-dir phase_c_runs/e04_outcome_L4_H128_L4_d2_N100k_eval
```

结果中的 `test_edge_reordered` 与 `test` 对比：若 free-run 指标明显下降，说明模型可能利用了边的序列顺序，不能把结果直接解释为稳定的图规划能力。

## 5. E04 指标解释与最小验收

`branching_reference_bits=L*log2(d)` 只是分支参考量，用于记录配置；它不是条件熵下界，也不与参数记忆容量放在同一个账本中。E04 不输出 `lambda` 或 `memory_bits`。

| 任务 | 主指标 | 对照基线 | 辅助指标 |
|---|---|---|---|
| outcome | `first_hop_accuracy` | `random_choice_accuracy=1/d` | teacher-forced NLL、`solver_em` |
| trace | `free_run_trace_em` | `random_choice_trace_em=(1/d)^(L-1)` | `stepwise_accuracy`、`first_error_position_mean`、`path_validity_rate`、`solver_em` |

一次配对实验至少检查：

1. 两个 run 的 `run_config.json` 除 `task`、`output_dir` 外训练定义相同。
2. 两者使用同一数据根目录及相同 unit 数；验证集只用于训练监控。
3. `solver_em` 应为 1.0；否则先检查数据生成或评估实现。
4. outcome 高于 `1/d` 才说明模型学到非随机首跳决策。
5. trace 同时报告 EM、逐位置准确率和合法路径率，不能只报告 teacher-forced NLL。
6. 边序重排后结果不能出现无法解释的大幅下降。

## 6. 训练前 Smoke Test

提交到服务器或改代码后，先在 CPU 跑完整回归：

```bash
python -m unittest discover -s phase_c/tests -p "test_*.py"
```

E04 极小 smoke：

```bash
python -m phase_c.cli dag train \
  --task outcome --model debug \
  --V 64 --L 3 --d 2 --W 4 \
  --train-size 20 --validation-size 10 --test-size 10 \
  --device cpu --dtype float32 \
  --micro-batch-size 1 --gradient-accumulation 1 \
  --max-steps 1 --eval-size 0 --eval-interval 0 \
  --output-dir phase_c_runs/e04_smoke_outcome
```

该 smoke 只验证数据、训练、free-run 评估和结果写盘，不能用于判断模型能力。

## 7. 常用排查

- `--task` 缺失：E04 必须明确指定 `outcome` 或 `trace`。
- `dataset-root requires ... units`：文件数据训练必须同时提供 train、validation、test 的 unit 数。
- `DAG unit does not match manifest`：数据文件被覆盖、截断或混用了另一个数据根目录；重新生成该目录。
- `final_metrics.json` 缺失：训练被中断；用 `dag eval --run-dir ...` 从最新 checkpoint 补评估。
- CUDA RNG 设备数报错：新代码只恢复当前可见 GPU 的 RNG state；请确认服务器使用的是更新后的仓库。
- 多卡未启动：必须使用 `torchrun --nproc_per_node=<卡数>`，不能只给普通 `python` 设置多个 `CUDA_VISIBLE_DEVICES`。

## 8. 推送前检查

```bash
python -m unittest discover -s phase_c/tests -p "test_*.py"
git status
```

不要执行 `git add .`，避免提交 `phase_c_dag_data/`、`phase_c_runs/` 或 checkpoint。
