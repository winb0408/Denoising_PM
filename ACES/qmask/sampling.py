"""QMask 采样器：共位相位分裂 + 轴参与掩码。

与 ACES 原 sampler 的关键差异：
    - 原 ellipsoid/empirical_block 把 4 视图映射成大步长不同坐标切片，
      N2N“同一内容两份噪声”前提被破坏；
    - QMask 始终在同一个 coarse 网格上做 phase split：参与轴在 2x2 相位内
      取子集，关闭轴 phase 固定 0（等价于该轴不下采样、原样保留）。因此
      VALID(三轴全开) 与 SN2N(Z 关、XY 开) 是同一掩码的两个端点。

输入 img 为 [N, Z, Y, X]；输出 4 个同设备视图，并兼容
``aces.sampling_aniso.SamplerMetadata``。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from aces.sampling_aniso import SamplerMetadata

# VALID 2x2x2 space-to-depth 的 channel 顺序是 z*4 + y*2 + x。
# 8 个角的下标 -> (dz, dy, dx)
CORNER_XYZ = [
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
]

# VALID generate_mask_pair 里的 8 行 idx_pair，用于三轴全开时随机选 4 角。
TETRA_ROWS = [
    [0, 1, 2, 4],
    [0, 1, 3, 5],
    [1, 2, 3, 7],
    [0, 2, 3, 6],
    [0, 4, 5, 6],
    [1, 4, 5, 7],
    [2, 4, 6, 7],
    [3, 5, 6, 7],
]

AXIS_ORDER = ("z", "y", "x")


def space_to_depth_multi(img: torch.Tensor, block_zyx: tuple[int, int, int]) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """按每轴 block size 做 space-to-depth，允许某轴 block=1（保留相位）。"""
    n, z, y, x = img.shape
    bz, by, bx = block_zyx
    z_pad = (-z) % bz
    y_pad = (-y) % by
    x_pad = (-x) % bx
    if z_pad or y_pad or x_pad:
        img = F.pad(img, (0, x_pad, 0, y_pad, 0, z_pad))
    n, z, y, x = img.shape
    out = img.view(n, z // bz, bz, y // by, by, x // bx, bx)
    out = out.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
    out = out.view(n, z // bz, y // by, x // bx, bz * by * bx)
    return out, (z_pad, y_pad, x_pad)


def _corner_index(phase_zyx: tuple[int, int, int], block_zyx: tuple[int, int, int]) -> int:
    bz, by, bx = block_zyx
    pz, py, px = phase_zyx
    return (pz * by + py) * bx + px


def _allowed_corners(participate: dict[str, bool]) -> tuple[list[int], tuple[int, int, int]]:
    """由轴参与掩码构造允许角集与每轴 block size。

    关轴 phase 固定 0、block=1（不下采样保留该轴）；开轴 phase {0,1}、block=2。
    """
    block = tuple(2 if participate[axis] else 1 for axis in AXIS_ORDER)
    ranges = []
    for axis in AXIS_ORDER:
        if participate[axis]:
            ranges.append([0, 1])
        else:
            ranges.append([0])
    corners: list[int] = []
    for pz in ranges[0]:
        for py in ranges[1]:
            for px in ranges[2]:
                corners.append(_corner_index((pz, py, px), block))
    return corners, block  # type: ignore[return-value]


def qmask_sampler(
    img: torch.Tensor,
    geometry,
    seed: int | None = None,
    row_mode: str = "fixed",
) -> tuple[list[torch.Tensor], SamplerMetadata]:
    if img.dim() != 4:
        raise ValueError(f"QMask sampler input must be [N,Z,Y,X], got {tuple(img.shape)}")
    if min(img.shape[1:]) < 2:
        raise ValueError(f"3D fail-fast: sampler input collapsed, got {tuple(img.shape)}")

    participate = geometry.effective_participate()
    on_count = sum(1 for axis in AXIS_ORDER if participate.get(axis))
    if on_count < 2:
        # 退化：几何层应已把参与轴提升到 >= 2；这里再兜底为全轴。
        participate = {axis: True for axis in AXIS_ORDER}
        note_note = "degenerate fallback to all-axes"
    else:
        note_note = "co-located phase split"

    corners, block = _allowed_corners(participate)

    if len(corners) == 8 and row_mode == "mosaic":
        # Q-3 消融变体：逐位置随机 Tetris 行（复刻 VALID generate_mask_pair 的
        # per-position 随机性），每个视图是随机相位马赛克而非固定相位网格。
        # 仅 8 角（全轴参与）时可用；与 fixed 的唯一差异是行选择粒度。        n = img.shape[0]
        sz, sy, sx = img.shape[1] // block[0], img.shape[2] // block[1], img.shape[3] // block[2]
        positions = n * sz * sy * sx
        gen = torch.Generator(device=img.device)
        gen.manual_seed(int(seed or 0))
        rd = torch.randint(0, len(TETRA_ROWS), (positions,), generator=gen, device=img.device)
        rows = torch.tensor(TETRA_ROWS, dtype=torch.long, device=img.device)[rd]  # [P,4]
        # 行内槽位随机排列（复刻 VALID RandomSampler_0123）：无此步则 (out,target)
        # 的相对偏移方向固定，模型可学"相位条件化平移"而非去噪（Q-3 第一轮教训）。
        perm = torch.randperm(4, generator=gen, device=img.device)
        s2d, _ = space_to_depth_multi(img, block)
        s2d_flat = s2d.reshape(positions, 8)
        arange = torch.arange(positions, device=img.device)
        views = [s2d_flat[arange, rows[:, perm[k]]].reshape(n, sz, sy, sx) for k in range(4)]
        chosen = TETRA_ROWS[int(seed or 0) % len(TETRA_ROWS)]
        return views, SamplerMetadata(
            mode="qmask",
            input_shape_nzyx=list(img.shape),
            output_shape_nzyx=list(views[0].shape),
            offsets_zyx=[[idx // (block[1] * block[2]), (idx % (block[1] * block[2])) // block[2], idx % block[2]] for idx in chosen],
            radii_px=None,
            pairability_threshold=float(geometry.tau_q),
            note=f"{note_note}; row_mode=mosaic (per-position random Tetris rows, VALID-like); block_zyx={block}",
        )

    if len(corners) == 8:
        row = TETRA_ROWS[int(seed or 0) % len(TETRA_ROWS)]
        chosen = [c for c in row if c in corners]  # row 在允许集内的角
        if len(chosen) != 4:
            chosen = TETRA_ROWS[0]  # 兜底固定行
    elif len(corners) == 4:
        chosen = sorted(corners)
    elif len(corners) == 2:
        # 不应发生（on_count>=2）；兜底为全轴。
        participate = {axis: True for axis in AXIS_ORDER}
        corners, block = _allowed_corners(participate)
        chosen = TETRA_ROWS[0]
    else:
        raise ValueError(f"Unexpected corner count {len(corners)}")

    if row_mode == "permuted" and len(chosen) == 4:
        # Q-3 修复（Plan v2 §2.2）：行内槽位随机排列，逐 batch 变化。
        # 固定的 view→corner 映射使 (out,target) 的相对偏移方向恒定，
        # 模型可学"相位条件化平移"而非去噪（3p 消融实测该修复值 +0.80 dB）。
        gen_slot = torch.Generator()  # CPU 生成器即可：只产生 4 个槽位的排列索引
        gen_slot.manual_seed(int(seed or 0))
        slot_perm = torch.randperm(4, generator=gen_slot).tolist()
        chosen = [chosen[i] for i in slot_perm]

    s2d, _ = space_to_depth_multi(img, block)  # type: ignore[arg-type]
    views = [s2d[..., idx] for idx in chosen]
    bz, by, bx = block
    offsets = [[idx // (by * bx), (idx % (by * bx)) // bx, idx % bx] for idx in chosen]
    return views, SamplerMetadata(
        mode="qmask",
        input_shape_nzyx=list(img.shape),
        output_shape_nzyx=list(views[0].shape),
        offsets_zyx=offsets,
        radii_px=None,
        pairability_threshold=float(geometry.tau_q),
        note=f"{note_note}; participate={participate}; block_zyx={block}; corners={chosen}",
    )
