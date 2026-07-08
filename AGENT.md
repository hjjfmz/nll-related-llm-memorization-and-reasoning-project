# Agent Guide

本文件约束在 `E:\keyan\resources` 中工作的 agent。项目当前围绕 LLM 推理深度与记忆容量的 Phase C 实验展开，核心代码在根目录，研究记录主要在 `week1/`、`week2/`、`week3/`。

## 项目定位

- 研究主题：用受控随机序列与分层 DAG 路径任务，测量模型的记忆容量、路径推理能力和深度惩罚。
- 当前主线：Phase C synthetic data -> admission checks -> from-scratch training -> capacity/reasoning measurement。
- 重要边界：预训练模型只能作为补充讨论；当前主证据路径是从零训练的受控实验。

## 主要文件

- `phase_c_data.py`：Phase C 数据生成、解析、校验和准入检查的核心实现。
- `generate_phase_c_data.py`：数据预览、准入验证、写出 `.jsonl.gz` shard 的 CLI。
- `phase_c_model.py`：decoder-only Transformer 及模型规模 preset。
- `phase_c_training.py`：answer-only collation、容量指标、流式样本、checkpoint 保存和恢复。
- `train_phase_c_random.py`：随机序列容量实验训练入口。除非用户明确批准，不要自动执行训练。
- `test_phase_c_data.py`、`test_phase_c_training.py`：当前轻量回归测试。
- `phase_c_outputs/`：已生成或验证过的输出。新增结果必须清楚标注阶段和配置。

## 工作原则

1. 先确认阶段，再动手。
   - 数据生成、准入检查、模型训练、结果测量、论文写作是不同阶段。
   - 用户只问某一阶段时，不要顺手推进到下一阶段。

2. 不要擅自启动训练。
   - 可以写训练代码、修训练脚本、做静态检查或小规模 CPU smoke test。
   - 不要自动启动 GPU/长时间训练，不要下载预训练权重，除非用户明确要求。

3. 结果必须按来源标注。
   - 明确区分：设计推导、代码生成、语法检查、单元测试、数据准入、CPU smoke test、实际训练结果。
   - 不要把 admission success 说成训练已经完成。

4. 保持实现窄而可验证。
   - 优先沿用现有脚本、参数名和数据格式。
   - 不做无关重构，不写大量额外说明文档，除非用户要求。
   - 修改共享数据格式、模型接口或训练指标时，必须同步检查相关测试。

5. 中文优先。
   - 用户通常用中文交流；解释、计划和阶段报告默认中文。
   - 论文英文段落或代码注释按具体任务需要决定。

## Phase C 数据约定

默认参数和既有选择：

- `V=1024`
- `S=384`
- `q=4`
- `N_train` 网格从 `100k` 起，可扩展到 `250k/500k/1M`
- `N_val=10k`
- `N_test=20k`
- `L={2,4,6,8}`
- `d={2,4}`
- `W=d+2`
- `b_i=1`

DAG 序列格式：

```text
[BOS] [GRAPH] edge pairs [QUERY] s t [ANSWER] path [EOS]
```

其中 `[GRAPH]` 后每两个 token 表示一条有向边 `u -> v`；答案路径省略重复的 source，只输出从 source 之后到 target 的节点序列。

准入检查必须关注：

- DAG 拓扑和唯一 `s -> t` 路径。
- 路径长度、出度、层结构是否符合配置。
- token 序列是否可解析回图和答案。
- train/validation/test 是否无重复样本或泄漏。
- `H_R_bits`、`H_L_bits`、长度统计、节点/边统计和词频平衡诊断。

## 常用命令

运行单元测试：

```powershell
python -m unittest test_phase_c_data.py test_phase_c_training.py
```

预览一个 DAG 样本：

```powershell
python generate_phase_c_data.py preview --family dag --V 128 --L 2 --d 2 --W 4 --sample-id 3 --seed 7
```

小规模准入验证：

```powershell
python generate_phase_c_data.py validate --family both --train-size 100 --validation-size 50 --test-size 50 --output-dir phase_c_outputs/admission_smoke
```

Windows 注意事项：

- 如果 `conda run` 对短参数解析异常，优先直接调用目标环境的 `python.exe`。
- 涉及中文输出或 JSON 时，优先使用 UTF-8 编码；Python 子进程可按需加 `-X utf8`。

## 交付前检查

提交回答前至少确认：

- 是否写入了用户要求的文件或代码。
- 是否没有越过用户给定阶段边界。
- 是否没有启动未经批准的训练或长任务。
- 是否说明了已运行和未运行的验证。
- 如果产生新输出，是否写清楚配置、seed、split、样本规模和结果阶段。

