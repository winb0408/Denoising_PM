"""Micro-probe 引擎（修正方案 M2：U_probe 校准）。

模块：
    - ``calibration.family_sampler``：CandidateFamily → 4 个共享 coarse grid
      的训练视图（``sample_views``）；
    - ``calibration.micro_probe``：极低预算训练探针（镜像 t4b_rgrid 协议），
      输出每 family 的 φ(Γ) 特征、robust z-score 与 U_probe 组合分
      （``MicroProbeRunner`` / ``MicroProbeConfig`` / CLI）。

注意：这里不 eager import 子模块，避免 ``python -m calibration.micro_probe``
触发 runpy 的 "found in sys.modules" 警告。
"""
