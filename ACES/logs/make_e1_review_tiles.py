"""E1 外部基线定性验收切块：noisy(8bit) | SN2N pred(尺度对齐) | target(proxy)。

每个 leg 产出一张 overview(≤1200px) + 若干行块(≤1200px, 远小于 3.5MB 上限)，
落盘到 /data3/wjb/Denoising_PM/Results/<leg>/review/。
遵循 CLAUDE.md 大图规则：验收只读 review 切块，绝不直接读原图。
"""
import os
import numpy as np
import tifffile

OUT_ROOT = '/data3/wjb/Denoising_PM/Results'

LEGS = {
    'sn2n_original_3p': {
        'noisy': OUT_ROOT + '/sn2n_original_3p/raw_data_8bit/3P_neuron_repeat01_275z_8bit.tif',
        'pred': OUT_ROOT + '/sn2n_original_3p/predictions/3P_neuron_repeat01_275z_8bit.tif_model_9_3_full.pth.tif',
        'target': '/data2/wjb/Denoising_PM/Methods/FAST-main/reproduction/valid_comparison/experiments/valid_paper_3p_neuron/original_workflow_paper_params_repeat01_seed3407/data/gt_proxy/3P_neuron_mean9_gt_proxy_275z.tif',
        'roi': None,  # 3P 用 target 分位生成前景 mask（同 E2 协议 p75 + 形态学）
        'a': 131.198668662158, 'b': 871.1020115804515,
    },
    'sn2n_original_neurofinder': {
        'noisy': OUT_ROOT + '/sn2n_original_neurofinder/raw_data_8bit/neurofinder_00_00_stack_8bit.tif',
        'pred': OUT_ROOT + '/sn2n_original_neurofinder/predictions/neurofinder_00_00_stack_8bit.tif_model_9_3_full.pth.tif',
        'target': '/data2/wjb/Denoising_PM/ACES/data/neurofinder.00.00/proxy/neurofinder_00_00_temporal_mean.tif',
        'roi': '/data2/wjb/Denoising_PM/ACES/data/neurofinder.00.00/proxy/neurofinder_00_00_roi_mask.tif',
        'a': 7.5770, 'b': 24.67,
    },
}

TILE = 1200


def norm_u8(a, lo_p=0.5, hi_p=99.5):
    lo, hi = np.percentile(a, [lo_p, hi_p])
    scale = max(float(hi - lo), 1e-6)
    return (np.clip((a - float(lo)) / scale, 0.0, 1.0) * 255.0).astype(np.uint8)


def save(path, arr):
    # 强制压到 ≤1100px（CLAUDE.md 上限 1200px/3.5MB，留余量）
    from skimage.transform import resize
    h, w = arr.shape[:2]
    scale = min(1.0, 1100.0 / max(h, w))
    if scale < 1.0:
        arr = resize(arr, (int(round(h * scale)), int(round(w * scale))),
                     order=1, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    tifffile.imwrite(path, arr)
    sz = os.path.getsize(path) / 1e6
    print(f'  wrote {path} shape={arr.shape} {sz:.2f} MB')
    assert arr.shape[0] <= 1200 and arr.shape[1] <= 1200, 'tile too large'
    assert sz < 3.5, 'tile > 3.5MB'


for leg, cfg in LEGS.items():
    print(f'== {leg} ==')
    rev = os.path.join(OUT_ROOT, leg, 'review')
    os.makedirs(rev, exist_ok=True)

    noisy = tifffile.imread(cfg['noisy']).astype(np.float32)
    pred = tifffile.imread(cfg['pred']).astype(np.float32)
    target = tifffile.imread(cfg['target']).astype(np.float32)
    print(' noisy', noisy.shape, 'pred', pred.shape, 'target', target.shape)

    pred_aligned = cfg['a'] * pred + cfg['b']

    # 对齐尺寸：nf 的 z 是时间轴，target 2D 沿 z 广播
    z = min(noisy.shape[0], pred.shape[0])
    y = min(noisy.shape[1], target.shape[-2])
    x = min(noisy.shape[2], target.shape[-1])
    noisy, pred_aligned = noisy[:z, :y, :x], pred_aligned[:z, :y, :x]
    if target.ndim == 2:
        tgt3 = np.broadcast_to(target[:y, :x], (z, y, x)).astype(np.float32)
        roi3 = None
        if cfg['roi']:
            roi = tifffile.imread(cfg['roi']).astype(bool)
            roi3 = np.broadcast_to(roi[:y, :x], (z, y, x))
    else:
        tgt3 = target[:z, :y, :x]
        roi3 = None

    # 前景 mask：nf 用 ROI；3P 用 target p75 + 形态学（同 patch_eval 协议）
    if roi3 is not None:
        fg = roi3
    else:
        from scipy import ndimage
        t2 = tgt3[0]
        thr = np.percentile(t2, 75)
        m = ndimage.binary_opening(t2 > thr, iterations=1)
        m = ndimage.binary_closing(m, iterations=1)
        fg = np.broadcast_to(m, (z, y, x))

    # 选前景最丰富的 z 中心位置（对 2D 广播 mask 恒定，取中段 z 即可）
    counts = [(int(fg[i:i+16].sum()), i) for i in range(0, z - 16, max(1, z // 40))]
    z0 = max(counts)[1]

    # 在该 z 带内找前景密度最高的 2D 窗口（512 或 275 大小内取 TILE/2 视窗）
    win = min(y, x, TILE // 2)
    band_fg = fg[z0:z0+16]
    best, by0, bx0 = -1.0, 0, 0
    step = max(1, win // 8)
    for yy in range(0, y - win + 1, step):
        for xx in range(0, x - win + 1, step):
            c = float(band_fg[:, yy:yy+win, xx:xx+win].mean())
            if c > best:
                best, by0, bx0 = c, yy, xx
    print(f' fg window at z0={z0} y0={by0} x0={bx0} fg_frac={best:.3f}')

    sl_noisy = noisy[z0:z0+16, by0:by0+win, bx0:bx0+win]
    sl_pred = pred_aligned[z0:z0+16, by0:by0+win, bx0:bx0+win]
    sl_tgt = tgt3[z0:z0+16, by0:by0+win, bx0:bx0+win]

    # 行块：每行 = 3 列(noisy/pred/target) 拼横图，取该带 3 个 z 切片
    zs = [z0, z0 + 8, z0 + 15]
    for ri, zi in enumerate(zs):
        cols = [norm_u8(sl_noisy[zi - z0]), norm_u8(sl_pred[zi - z0]), norm_u8(sl_tgt[zi - z0])]
        row = np.concatenate(cols, axis=1)
        save(os.path.join(rev, f'row_{ri}_z{zi}.png'), row)

    # overview：带内中位 z 切片（带内索引），3 列拼图（save() 会自动压到 ≤1100px）
    zmid = 8
    cols = [norm_u8(v[zmid]) for v in (sl_noisy, sl_pred, sl_tgt)]
    ov = np.concatenate(cols, axis=1)
    save(os.path.join(rev, 'qualitative_overview.png'), ov)
    # 额外：最大强度投影（z 带内），看整体结构有无塌缩
    mips = [norm_u8(v.max(axis=0)) for v in (sl_noisy, sl_pred, sl_tgt)]
    ov_mip = np.concatenate(mips, axis=1)
    save(os.path.join(rev, 'mip_overview.png'), ov_mip)
print('done')
