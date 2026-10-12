"""ACES profiling：pair-risk field 引擎（修正方案 Milestone 1）。

模块：
  - ``candidates``：候选 offset / sampling family 库（§7）；
  - ``risk``：R_N / R_S / R_I 三个风险估计器（§8-§10）；
  - ``risk_field``：CLI，产出逐 offset 与 family 级风险表（Experiment B）。
"""

from .candidates import (
    CandidateFamily,
    CandidateOffset,
    default_family_library,
    default_offset_library,
)
from .risk import (
    NoiseRiskEstimator,
    ShortcutRiskEstimator,
    StructureRiskEstimator,
    shifted_pair,
)

__all__ = [
    "CandidateFamily",
    "CandidateOffset",
    "default_family_library",
    "default_offset_library",
    "NoiseRiskEstimator",
    "ShortcutRiskEstimator",
    "StructureRiskEstimator",
    "shifted_pair",
]
