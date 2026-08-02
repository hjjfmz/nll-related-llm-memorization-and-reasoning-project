"""Experiment 5 (E5): Unified ledger analysis. 骨架占位，待实现。

对应 project.md 第五关「统一账本分析」与 idea_report 实验五。

前置条件：
- E3 已测得可靠容量密度 ``α``（``C_phys ≈ αP``）
- E4 已测得相对稳定的深度惩罚 ``λ``
- DAG 新图测试显著高于随机基线

目标：检验统一账本假设——任务代价 ``H_L + λL`` 超过容量 ``αP`` 时
推理崩塌，并预测崩塌深度：

    collapse expectation:  H_L + λL ≈ αP

待实现模块：
- ``capacity.py``：汇总 E3/E4 的 ``final_metrics.json``
- ``collapse.py``：拟合 ``M_max(P) = αP + c``、估计 ``λ``、预测崩塌深度
- ``reporting.py``：λ 稳定性图、αP 与 DAG 崩塌位置关系图
"""
