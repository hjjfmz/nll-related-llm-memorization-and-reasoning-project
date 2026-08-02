"""Experiment 4 (E4): DAG reasoning limit measurement. 骨架占位，待实现。

对应 project.md 第四关「纯 DAG 推理极限测量」与 idea_report 实验四。

目标：测模型在 IID 新图上的路径推理，而非训练图记忆。

约定（源自 phase_c.data.DagConfig 冻结 contract）：
- 序列格式：``[BOS] [GRAPH] 边对序列 [QUERY] s t [ANSWER] 路径 [EOS]``
- 分层 DAG，每层 W 个节点（默认 W=d+2），主链唯一，``b_i=1``
- 逻辑熵 ``H_L = L * log2(d)`` 先验可算

待实现模块：
- ``train.py``：纯 DAG（family=dag）训练入口
- ``config.py``：``(L, d)`` 网格与 run 参数构造
- ``evaluation.py``：NLL、Exact Match、stepwise accuracy、
  first-error position、path validity、随机基线、确定性求解器上界
- ``reporting.py``：``NLL/EM/λ vs L`` 等图

关键指标：``λ = (NLL_test - H_L) / L``，对角线设计
（train depth = test depth）消除训练浅/测试深的泛化偏差。
"""
