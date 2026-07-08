我会把它整理成“下一步执行路线 + 每一步验收标准 + 结果分析口径”。这里最重要的是别把 smoke test、准入检查、正式训练结果混在一起，否则后面论文叙事会很容易跑偏。


我建议接下来不要一上来做“大实验”，而是按“准入关卡 → smoke 关卡 → 随机容量正式标定 → DAG 推理正式测量 → 统一账本分析”的顺序推进。每一关只回答一个问题，验收通过再进入下一关。
总路线
当前最稳的实验路线是：
实验 0：冻结数据 contract
确认当前生成器、序列格式、metadata、准入报告字段不再漂移。

实验 1：数据与测量管线准入
验证随机序列和 DAG 数据真的干净、可解析、无捷径。

实验 2：训练脚本 CPU smoke test
只验证训练代码能跑通，不解释为实验结果。

实验 3：纯随机序列容量标定
得到本项目自己的 αP，即模型在当前架构和训练协议下能记多少随机 bit。

实验 4：纯 DAG 推理极限测量
在 IID 新图上测 NLL / EM / stepwise accuracy / λ，判断是否学到可迁移路径推理。

实验 5：统一账本分析
只有随机容量 αP 和 DAG 深度惩罚 λ 都稳定后，才分析二者能否共同预测崩塌位置。

第一关：数据 Contract 与准入
目标：确认数据本身可以进入训练，而不是确认模型能力。
要做：
读取并固定 [phase_c_data.py](E:/keyan/resources/phase_c_data.py)、[generate_phase_c_data.py](E:/keyan/resources/generate_phase_c_data.py)、[test_phase_c_data.py](E:/keyan/resources/test_phase_c_data.py) 的当前约定。
小规模跑一次 family=both 的 validation。
对已有 phase_c_outputs/admission_100k_L4_d2/admission_report.json 做结构复核。
验收内容：
all_checks_passed = true
random / dag 数量与 split 数量正确。
train / validation / test 无重复 key、target、graph hash。
random 的 H_R = S log2 V 正确。
DAG 的 H_L = L log2 d 正确，但只作为结构难度锚点。
DAG 每张图满足：有向无环；
严格分层；
唯一 s -> t 路径；
路径长度等于 L；
非终层出度等于 d；
token 序列可逆解析；
边序打乱不改变答案。

节点词频、path/non-path 频率无明显系统偏差。
分析输出：
一张“数据准入表”：配置：V/S/q/L/d/W/N_train/N_val/N_test/seed
统计：长度、节点数、边数、H_R_bits、H_L_bits
检查：pass/fail
泄漏：duplicate counts
词频：max z-score

这一关通过后，才能说“数据可用”。还不能说“训练成功”。
第二关：训练脚本 CPU Smoke Test
目标：确认训练脚本、collator、loss、checkpoint、metrics 没断。它不是论文结果。
要做：
只用极小模型、极小数据、极少 step、CPU。
优先检查 [train_phase_c_random.py](E:/keyan/resources/train_phase_c_random.py)、[phase_c_training.py](E:/keyan/resources/phase_c_training.py) 是否支持：CPU；
小 batch；
小 step；
--max-steps 或类似参数；
checkpoint save/load；
metrics 写出。

如果缺少 smoke 参数，再加，不改变正式训练默认值。
验收内容：
脚本能启动、跑完、正常退出。
loss 是有限值，不是 NaN/inf。
answer-only label mask 正确，只算答案 token。
随机序列的 answer_start/answer_end 对齐，尤其不要把 prefix 或 EOS 算进容量。
checkpoint 能保存。
如有 resume，resume 后 step、loss、optimizer state 不乱。
metrics 文件包含至少：train NLL total；
train NLL per token；
validation NLL per token；
M_train 或计算 M_train 所需字段；
config 和 seed。

分析输出：
只报告：“训练脚本 smoke test 通过/失败”。
不报告 bits/parameter。
不画容量曲线。
不把 smoke loss 当作模型能力。
第三关：纯随机序列容量标定
目标：得到本项目当前模型族和训练协议下的 M_max(P)=αP+c。
这是正式训练的第一阶段，需要用户明确同意后再跑。
要做：
只训练 family=random。
多个模型规模 P。
每个 P 扫多个 N_train，覆盖：欠容量区；
过渡区；
饱和区。

验证集只用于不可泛化性检查，不用于计算容量。
核心指标：
H_R = S log2 V

M_train = N_train * H_R - NLL_train,total

bits_per_param = M_train / P
验收内容：
训练集 NLL 随训练下降。
M_train 随训练上升。
未见随机序列的 per-token NLL 接近 log2 V。
M_train 随 N_train 增大出现饱和。
同一 P 的多个 seed 差异小于规模效应。
不同 P 的 M_max(P) 单调增长。
线性拟合 M_max(P)=αP+c 在测量范围内合理。
分析内容：
图 1：每个 P 的容量曲线
横轴 N_train * H_R，纵轴 M_train。
图 2：容量缩放律
横轴 P，纵轴 M_max，拟合 αP+c。
图 3：训练集 vs 未见随机序列 NLL
证明训练集可记忆，未见随机不可泛化。
表格：exact command；
seed；
model preset；
V/S/q/N_train/N_val/N_test；
checkpoint；
train NLL total；
train NLL/token；
val NLL/token；
M_train；
bits/parameter。

这一关通过后，才能说“本项目内随机记忆容量已标定”。
第四关：纯 DAG 推理极限测量
目标：测模型在未见新图上的路径推理，而不是训练图记忆。
要做：
只用 family=dag。
不混入随机序列。
先做“对角线”条件：train depth = test depth；
不做浅层训练、深层测试；
这样避免把泛化 gap 混进 λ。

固定 d 和 P，扫 L。
再换 d、换 P。
每组训练到 validation 平台后，再测 IID 新图 test。
核心指标：
H_L = L log2 d

lambda = (NLL_test - H_L) / L
同时记录：
test NLL in bits；
Exact Match；
conditional stepwise accuracy；
first-error position；
path validity rate；
random-choice baseline；
deterministic solver upper bound；
input token length；
graph node/edge count；
context window occupancy。
验收内容：
模型能被认为“学到可迁移 DAG 推理”，至少需要：
IID 新图性能显著高于随机选择基线。
确定性图搜索程序在同批数据上 100% 正确。
节点重命名后性能不明显变化。
边序重排后性能不明显变化。
删图对照性能接近随机。
错图对照性能明显下降。
输出路径合法率高。
新图表现不能只在训练图上成立。
分析内容：
图 1：L vs test NLL。
图 2：L vs Exact Match。
图 3：路径位置 vs conditional stepwise accuracy。
图 4：first-error position 分布。
图 5：L vs λ，按 P 或 d 分组。
图 6：扰动与负对照表：原图；
重命名；
边序重排；
删图；
错图。

这一关通过后，才能说“模型在新图上有可迁移路径推理能力”。如果只在训练图上好，不能算。
第五关：统一账本分析
目标：在随机容量和 DAG 推理都稳定后，再讨论是否存在统一容量解释。
前置条件：
实验三已经得到可靠 α。
实验四已经得到相对稳定的 λ。
DAG 新图测试显著高于随机 baseline。
控制实验排除了命名、边序、删图、错图捷径。
输入长度、图规模、上下文占用已经记录。
可以分析：
capacity side:
C_phys ≈ αP

task side:
cost_DAG ≈ H_L + λL

collapse expectation:
H_L + λL ≈ αP
验收内容：
用浅/中深度估计的 λ 能否预测更深配置的 NLL 或崩塌位置。
更大 P 是否把崩塌点推向更深 L 或更大 d。
预测方向是否与 αP 一致。
λ 是否在有效区间内稳定。
如果 λ 系统漂移，要报告线性基准失败，不要强行拟合统一定律。
分析输出：
预测崩塌深度 vs 实测崩塌深度。
λ 稳定性图。
αP 与 DAG 崩塌位置关系图。
Claim-Evidence Map 更新：哪些 claim 已支持；
哪些只是趋势；
哪些被证伪；
哪些还没跑。
