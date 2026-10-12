"""候选 offset / sampling family 库（修正方案 §7 Candidate Sampling Family Library）。

第一版不搜索无限 3D offset space，而是建立可控的候选库：

- Axis offsets：单轴 / 双轴 / 三轴参与，距离 1 和 2；
- Diagonal offsets：(1,1,0), (1,0,1), (0,1,1), (1,1,1) 及其互补相位
  （符号翻转与混合距离变体，如 (1,-1,0), (2,1,0)）；
- Family：一组 4 个 view 在共享 coarse grid 上的 offset δ_i，
  pair 层面的候选是所有 Δ_ij = δ_j − δ_i（i<j）。

物理空间（方案 §7.2）：每个 candidate 同时记录 voxel offset 与 spacing，
``offset_um = offset_voxel * spacing``，只有换到 um 空间才能讨论各向异性。
注意 z spacing 是根据数据布局推导的标定值（非原始设备 metadata）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# zyx 顺序的 voxel offset（与体数据的 numpy 轴顺序一致）
OffsetZyx = tuple[int, int, int]
SpacingUm = tuple[float, float, float]


@dataclass(frozen=True)
class CandidateOffset:
    """单个候选 pair offset（voxel 空间 + 可选物理 spacing）。"""

    offset_zyx: OffsetZyx
    spacing_um: SpacingUm | None = None

    @property
    def offset_um(self) -> SpacingUm | None:
        """物理位移 Δ_um = (Δz·sz, Δy·sy, Δx·sx)；spacing 缺失时返回 None。"""
        if self.spacing_um is None:
            return None
        return tuple(
            float(offset * spacing)
            for offset, spacing in zip(self.offset_zyx, self.spacing_um)
        )


@dataclass
class CandidateFamily:
    """一个 sampling family：4 个 view 在共享 coarse grid 上的 offset。

    family 的 pair 风险由所有 pairwise offset Δ_ij = δ_j − δ_i（i<j）聚合
    （mean / worst-case，见方案 §6 family-level risk）。
    """

    family_id: str
    offsets_zyx: list[OffsetZyx] = field(default_factory=list)

    def pairwise_offsets(self) -> list[OffsetZyx]:
        """返回去重后的全部 Δ_ij = δ_j − δ_i（i<j，保留符号）。

        注意 Δ_ij 与 Δ_ji 互为相反数；对平稳场二者风险近似相等，
        这里只保留 i<j 的一侧（12 对，若存在重复则更少）。
        """
        deltas: list[OffsetZyx] = []
        for i in range(len(self.offsets_zyx)):
            for j in range(i + 1, len(self.offsets_zyx)):
                delta = tuple(
                    self.offsets_zyx[j][axis] - self.offsets_zyx[i][axis]
                    for axis in range(3)
                )
                if delta not in deltas:
                    deltas.append(delta)  # type: ignore[arg-type]
        return deltas


def _dedup(offsets: list[OffsetZyx]) -> list[OffsetZyx]:
    """按出现顺序去重。"""
    seen: list[OffsetZyx] = []
    for offset in offsets:
        if offset not in seen:
            seen.append(offset)
    return seen


def default_offset_library(
    spacing_um: SpacingUm | None = None,
) -> list[CandidateOffset]:
    """默认 offset 候选库（~27 个唯一 offset）。

    覆盖三个层次（方案 §7.1）：
      A. Axis offsets：x/y/z 单轴、xy/xz/yz、xyz 组合，距离 1 与 2；
      B. Diagonal offsets：(1,1,0),(1,0,1),(0,1,1),(1,1,1) 及互补相位
         （符号翻转 (1,-1,0) 等）；
      C. Multi-distance / 混合距离变体：(2,1,0),(1,2,0),(2,2,1) 等。
    """
    axis_offsets: list[OffsetZyx] = [
        # A. 轴参与族，距离 1 / 2（zyx）
        (0, 0, 1), (0, 0, 2),          # x-only
        (0, 1, 0), (0, 2, 0),          # y-only
        (1, 0, 0), (2, 0, 0),          # z-only
        (0, 1, 1), (0, 2, 2),          # xy
        (1, 0, 1), (2, 0, 2),          # xz
        (1, 1, 0), (2, 2, 0),          # yz
        (1, 1, 1), (2, 2, 2),          # xyz
    ]
    diagonal_complements: list[OffsetZyx] = [
        # B. 对角互补相位（符号翻转：corner 的另一组对角方向）
        (1, -1, 0), (1, 0, -1), (0, 1, -1),
        # C. 混合距离变体（“互补 phase 组合”的推广）
        (2, 1, 0), (1, 2, 0), (2, 0, 1), (1, 0, 2), (0, 2, 1), (0, 1, 2),
        (2, 2, 1), (2, 1, 2), (1, 2, 2),
    ]
    offsets = _dedup(axis_offsets + diagonal_complements)
    return [CandidateOffset(offset, spacing_um=spacing_um) for offset in offsets]


def default_family_library() -> list[CandidateFamily]:
    """默认 sampling family 候选库。

    覆盖已有方法的几何端点（方案 §6/§7）：
      - VALID-like：2×2×2 corner 划分（每轴 d=1 的单轴参与 corner）；
      - SN2N-like：xy 对角方向上等距的 4 个 view；
      - QMask-like：轴参与变体（xy / xz / yz / xyz block 的 4 个 corner）；
      - Axis / diagonal families：单轴与对角方向上距离递增的 4 个 view。
    每个 family 携带其 4 个 view 在 coarse grid 上的 offset。
    """
    families: list[CandidateFamily] = [
        # --- 单轴族：沿一个轴等距铺 4 个 view ---
        CandidateFamily("axis_x_d1", [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)]),
        CandidateFamily("axis_y_d1", [(0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0)]),
        CandidateFamily("axis_z_d1", [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]),
        CandidateFamily("axis_x_d2", [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)]),
        CandidateFamily("axis_y_d2", [(0, 0, 0), (0, 2, 0), (0, 4, 0), (0, 6, 0)]),
        CandidateFamily("axis_z_d2", [(0, 0, 0), (2, 0, 0), (4, 0, 0), (6, 0, 0)]),
        # --- VALID-like：2×2×2 的 4 个 corner（三轴各参与 d=1）---
        CandidateFamily("valid_corners", [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]),
        # --- QMask-like：轴参与 block 的 4 个 corner ---
        CandidateFamily("qmask_xy", [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]),
        CandidateFamily("qmask_xz", [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)]),
        CandidateFamily("qmask_yz", [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]),
        # xyz 参与：正四面体式 4 corner，pairwise 覆盖三条面对角线及其反号
        CandidateFamily("qmask_xyz_tetra", [(0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)]),
        # --- SN2N-like：xy 对角方向等距 4 view ---
        CandidateFamily("sn2n_xy_diag", [(0, 0, 0), (0, 1, 1), (0, 2, 2), (0, 3, 3)]),
        # --- 对角族 ---
        CandidateFamily("diag_xz", [(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 3)]),
        CandidateFamily("diag_yz", [(0, 0, 0), (1, 1, 0), (2, 2, 0), (3, 3, 0)]),
        CandidateFamily("diag_xyz", [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]),
    ]
    for family in families:
        if len(family.offsets_zyx) != 4:
            raise ValueError(f"family {family.family_id} 必须有 4 个 view offset")
    return families
