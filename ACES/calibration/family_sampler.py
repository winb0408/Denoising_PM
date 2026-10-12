"""Family 视图采样器（M2 micro-probe 的数据侧）。

把 :class:`profiling.candidates.CandidateFamily` 的 4 个 view offset δ_i
映射到共享 coarse grid 上（修正方案 §3 sampling family 定义）：

    view_i[c, iz, iy, ix] = P[c, δ_iz + iz·bz, δ_iy + iy·by, δ_ix + ix·bx]

其中 block（采样步长）B=(bz,by,bx) 由 family 的 offset 跨度推导：每轴
``b = max(δ) − min(δ) + 1``，offsets 归一化（减每轴 min）后落在 [0, b) 内。
当 4 个 offset 恰为某 block 的 4 个 corner（如 2×2×2 corner family）时，
本采样器与 ``qmask.sampling.space_to_depth_multi`` 的 phase-split 逐值等价
（仅通道排布不同）；一般小 offset 通过「按 δ_i 裁剪 + 共同 stride 切片」
实现，无需 padding，完全确定性（无 RNG）。

支持的 patch 形状：
    - 4 维 ``(C, Z, Y, X)``：通道在前（batch 维也可以当作 C 传入）；
    - 3 维 ``(Z, Y, X)``：无通道；2 维 ``(Y, X)``：无 z 轴（要求 family 的
      全部 δ_z = 0，即退化为 2D family）。
返回的视图保持输入的维度约定（4 维进 4 维出，以此类推）。
"""

from __future__ import annotations

from typing import Any

import torch

from profiling.candidates import CandidateFamily

# view 数固定为 4（与 CandidateFamily 库约定一致）
N_VIEWS = 4


def derive_block(
    family: CandidateFamily,
) -> tuple[tuple[int, int, int], list[tuple[int, int, int]]]:
    """由 family offsets 推导共享 coarse grid 的 block（stride）与归一化 offsets。

    每轴 ``b = max(δ) − min(δ) + 1``；offsets 减去每轴 min 后落在 [0, b)，
    保证 4 个视图覆盖同一组 coarse 位置（view_i(k) = P(δ_i + B·k)）。
    """
    if len(family.offsets_zyx) != N_VIEWS:
        raise ValueError(
            f"family {family.family_id} 必须有 {N_VIEWS} 个 view offset, "
            f"got {len(family.offsets_zyx)}"
        )
    offsets = family.offsets_zyx
    mins = [min(delta[axis] for delta in offsets) for axis in range(3)]
    maxs = [max(delta[axis] for delta in offsets) for axis in range(3)]
    block = tuple(maxs[axis] - mins[axis] + 1 for axis in range(3))
    normalized = [
        tuple(delta[axis] - mins[axis] for axis in range(3)) for delta in offsets
    ]
    return block, normalized  # type: ignore[return-value]


def sample_views(
    patch: torch.Tensor, family: CandidateFamily
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """从体 patch 构造 family 的 4 个训练视图（确定性，无 RNG）。

    Args:
        patch: ``(C,Z,Y,X)`` / ``(Z,Y,X)`` / ``(Y,X)`` float tensor。
        family: 4 个 view offset 的 sampling family（zyx）。

    Returns:
        (views, metadata)：views 为 4 个同形状 tensor；
        metadata 含 offsets / normalized_offsets / block(stride) / 输入输出形状。
    """
    if not torch.is_floating_point(patch):
        patch = patch.float()
    had_channel = patch.dim() == 4
    had_z = patch.dim() >= 3
    if patch.dim() == 4:
        tensor = patch
    elif patch.dim() == 3:
        tensor = patch.unsqueeze(0)  # (1,Z,Y,X)
    elif patch.dim() == 2:
        tensor = patch.reshape(1, 1, *patch.shape)  # (1,1,Y,X)
    else:
        raise ValueError(
            f"patch 维度必须是 2/3/4 (Y,X)/(Z,Y,X)/(C,Z,Y,X), got {tuple(patch.shape)}"
        )

    _, z_len, y_len, x_len = tensor.shape
    block, norm_offsets = derive_block(family)
    bz, by, bx = block

    # 2D patch（无 z 轴）要求 family 全部 δ_z = 0（归一化后 span_z = 0 → bz = 1）
    if not had_z and bz != 1:
        raise ValueError(
            f"2D patch 不支持带 z offset 的 family {family.family_id} (block={block})"
        )

    # 共享 coarse grid 的每轴站点数：Z' = Z // bz（归一化 offset δ ≤ b−1 时
    # δ + (Z'−1)·b + 1 ≤ Z'·b ≤ Z 恒成立，即每个视图都能完整取到 Z' 个点）
    n_z = z_len // bz
    n_y = y_len // by
    n_x = x_len // bx
    if min(n_z, n_y, n_x) < 1:
        raise ValueError(
            f"patch {tuple(tensor.shape)} 对 family {family.family_id} (block={block}) "
            f"太小: coarse 站点数 (z,y,x)=({n_z},{n_y},{n_x})"
        )

    views: list[torch.Tensor] = []
    for dz, dy, dx in norm_offsets:
        view = tensor[
            :,
            dz : dz + n_z * bz : bz,
            dy : dy + n_y * by : by,
            dx : dx + n_x * bx : bx,
        ]
        if had_channel:
            views.append(view)
        elif had_z:
            views.append(view.squeeze(0))
        else:
            views.append(view.squeeze(0).squeeze(0))

    metadata: dict[str, Any] = {
        "family_id": family.family_id,
        "offsets_zyx": [tuple(int(v) for v in delta) for delta in family.offsets_zyx],
        "normalized_offsets_zyx": [tuple(int(v) for v in delta) for delta in norm_offsets],
        "block_zyx": tuple(int(v) for v in block),
        "input_shape": list(patch.shape),
        "output_shape": list(views[0].shape),
    }
    return views, metadata
