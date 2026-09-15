#!/usr/bin/env python
"""E1 外部基线(SN2N 原版)定性代理数值审计。

背景:当前会话的 Read 工具返回 base64 而非视觉内容,肉眼判读不可用。
用数值代理替代"塌缩/伪影"目检,指标与判读阈值:

- std 比值 pred/target(global/fg/bg): <0.4 → 塌缩
- Pearson 相关(global/fg): <0.3 → 结构丢失
- SBR(fg/bg 均值比): pred 相对 target 的保持率
- 饱和/死像素率: >5% / >10% → 伪影
- 局部方差(块 std)图相关: <0 → 纹理结构反转
- uint8 卷绕: 原始输出是否含 0/255 极值堆积

输出: 每 leg 打印表格 + JSON 落盘 review/numeric_audit.json
"""
import json
import os

import numpy as np
import tifffile
from scipy import ndimage

OUT_ROOT = '/data3/wjb/Denoising_PM/Results'
MEAN9_PROXY = ('/data2/wjb/Denoising_PM/Methods/FAST-main/reproduction/'
               'valid_comparison/experiments/valid_paper_3p_neuron/'
               'original_workflow_paper_params_repeat01_seed3407/data/gt_proxy/'
               '3P_neuron_mean9_gt_proxy_275z.tif')

LEGS = {
    '3p': {
        'noisy': OUT_ROOT + '/sn2n_original_3p/raw_data_8bit/3P_neuron_repeat01_275z_8bit.tif',
        'pred': OUT_ROOT + '/sn2n_original_3p/predictions/3P_neuron_repeat01_275z_8bit.tif_model_9_3_full.pth.tif',
        'target': MEAN9_PROXY,
        'roi': None,  # 从 target 派生(与切图脚本一致)
        'a': 131.198668662158, 'b': 871.1020115804515,
    },
    'neurofinder': {
        'noisy': OUT_ROOT + '/sn2n_original_neurofinder/raw_data_8bit/neurofinder_00_00_stack_8bit.tif',
        'pred': OUT_ROOT + '/sn2n_original_neurofinder/predictions/neurofinder_00_00_stack_8bit.tif_model_9_3_full.pth.tif',
        'target': '/data2/wjb/Denoising_PM/ACES/data/neurofinder.00.00/proxy/neurofinder_00_00_temporal_mean.tif',
        'roi': '/data2/wjb/Denoising_PM/ACES/data/neurofinder.00.00/proxy/neurofinder_00_00_roi_mask.tif',
        'a': 7.5770, 'b': 24.67,
    },
}

BLOCK = 8   # 局部方差块大小(y/x)
CHUNK = 64  # z 分块


def accum(stats, region, x, y):
    """region: 'all'|'fg'|'bg'; x=pred_aligned, y=target (chunk)"""
    s = stats[region]
    x = x.ravel().astype(np.float64)
    y = y.ravel().astype(np.float64)
    s['n'] += x.size
    s['sx'] += x.sum(); s['sy'] += y.sum()
    s['sxx'] += (x * x).sum(); s['syy'] += (y * y).sum()
    s['sxy'] += (x * y).sum()


def finalize(stats):
    out = {}
    for r, s in stats.items():
        n = max(s['n'], 1)
        mx, my = s['sx'] / n, s['sy'] / n
        vx = max(s['sxx'] / n - mx * mx, 0.0)
        vy = max(s['syy'] / n - my * my, 0.0)
        cov = s['sxy'] / n - mx * my
        stdx, stdy = np.sqrt(vx), np.sqrt(vy)
        out[r] = {
            'mean': mx, 'std': stdx,
            'std_ratio': stdx / max(stdy, 1e-9),
            'pearson': cov / max(np.sqrt(vx * vy), 1e-12),
        }
    return out


def block_std(a, block=BLOCK):
    """2D 块 std 图(y/x 平面), a: (Y,X)"""
    H, W = (a.shape[0] // block) * block, (a.shape[1] // block) * block
    a = a[:H, :W].reshape(H // block, block, W // block, block)
    return a.std(axis=(1, 3))


def audit_leg(name, cfg):
    print(f'\n===== {name} =====')
    pred_raw = tifffile.imread(cfg['pred'])
    noisy = tifffile.imread(cfg['noisy']).astype(np.float32)
    target = tifffile.imread(cfg['target']).astype(np.float32)
    print(f'raw pred dtype={pred_raw.dtype} shape={pred_raw.shape} '
          f'min={pred_raw.min()} max={pred_raw.max()}')

    # uint8 极值堆积(卷绕/饱和, 用原始输出)
    pr = pred_raw.astype(np.float32)
    lo, hi = float(pr.min()), float(pr.max())
    hist_lo = float((pr <= lo + 0.5).mean())
    hist_hi = float((pr >= hi - 0.5).mean())

    pred_aligned = cfg['a'] * pr + cfg['b']
    z = min(noisy.shape[0], pred_aligned.shape[0])
    y = min(noisy.shape[1], target.shape[-2])
    x = min(noisy.shape[2], target.shape[-1])
    noisy, pred_aligned, pr = noisy[:z, :y, :x], pred_aligned[:z, :y, :x], pr[:z, :y, :x]

    if target.ndim == 2:
        tgt3 = np.broadcast_to(target[:y, :x], (z, y, x)).astype(np.float32)
        if cfg['roi'] is not None:
            roi2 = tifffile.imread(cfg['roi']).astype(bool)
            fg3 = np.broadcast_to(roi2[:y, :x], (z, y, x))
        else:
            fg3 = None
    else:
        tgt3 = target[:z, :y, :x]
        fg3 = None

    if fg3 is None:
        t2 = tgt3[0]
        thr = np.percentile(t2, 75)
        m = ndimage.binary_opening(t2 > thr, iterations=1)
        m = ndimage.binary_closing(m, iterations=1)
        fg3 = np.broadcast_to(m, (z, y, x))

    stats = {r: dict(n=0, sx=0.0, sy=0.0, sxx=0.0, syy=0.0, sxy=0.0)
             for r in ('all', 'fg', 'bg')}
    bg3 = ~fg3

    # 局部方差图相关: 沿 z 取块的 2D 块 std, 对应块内做相关
    lv_pred, lv_tgt = [], []
    dead_blocks = 0
    total_blocks = 0

    for z0 in range(0, z, CHUNK):
        z1 = min(z0 + CHUNK, z)
        pa, ta = pred_aligned[z0:z1], tgt3[z0:z1]
        accum(stats, 'all', pa, ta)
        accum(stats, 'fg', pa[fg3[z0:z1]], ta[fg3[z0:z1]])
        accum(stats, 'bg', pa[bg3[z0:z1]], ta[bg3[z0:z1]])
        # 局部方差(对 chunk 均值面做块 std, 避免逐帧噪声主导)
        lp = block_std(pa.mean(axis=0))
        lt = block_std(ta.mean(axis=0))
        lv_pred.append(lp); lv_tgt.append(lt)
        dead_blocks += int((lp < 1e-6).sum())
        total_blocks += lp.size

    res = finalize(stats)
    lv_p = np.concatenate([v.ravel() for v in lv_pred])
    lv_t = np.concatenate([v.ravel() for v in lv_tgt])
    lv_corr = float(np.corrcoef(lv_p, lv_t)[0, 1])

    # res 中 fg/bg 的 mean 都是 pred_aligned 的; target 的 fg/bg 均值比单独算
    sbr_pred = res['fg']['mean'] / max(res['bg']['mean'], 1e-9)
    mean_t_fg = float(tgt3[fg3].astype(np.float64).mean())
    mean_t_bg = float(tgt3[bg3].astype(np.float64).mean())
    sbr_target = mean_t_fg / max(mean_t_bg, 1e-9)

    out = {
        'shape': list(pred_aligned.shape),
        'raw_pred_minmax': [lo, hi],
        'sat_low_frac': hist_lo, 'sat_high_frac': hist_hi,
        'dead_block_frac': dead_blocks / max(total_blocks, 1),
        'local_var_corr': lv_corr,
        'std_ratio': {r: res[r]['std_ratio'] for r in res},
        'pearson': {r: res[r]['pearson'] for r in res},
        'mean_pred': {r: res[r]['mean'] for r in res},
        'sbr_target': sbr_target, 'sbr_pred_aligned': sbr_pred,
        'sbr_kept_frac': sbr_pred / max(sbr_target, 1e-9),
    }

    # 判读
    flags = []
    if out['std_ratio']['fg'] < 0.4:
        flags.append(f"COLLAPSE: fg std_ratio={out['std_ratio']['fg']:.2f} < 0.4")
    if out['pearson']['fg'] < 0.3:
        flags.append(f"STRUCT-LOSS: fg pearson={out['pearson']['fg']:.2f} < 0.3")
    if out['sat_low_frac'] > 0.05 or out['sat_high_frac'] > 0.05:
        flags.append(f"SATURATION: lo={out['sat_low_frac']:.3f} hi={out['sat_high_frac']:.3f} > 5%")
    if out['dead_block_frac'] > 0.10:
        flags.append(f"DEAD-BLOCKS: {out['dead_block_frac']:.2%} > 10%")
    if out['local_var_corr'] < 0.0:
        flags.append(f"TEXTURE-INVERT: local_var_corr={out['local_var_corr']:.2f} < 0")
    out['verdict'] = 'PASS (无塌缩/伪影数值证据)' if not flags else '; '.join(flags)

    for k, v in out.items():
        if k != 'shape':
            print(f'  {k}: {v}')

    rev = os.path.join(OUT_ROOT, f'sn2n_original_{name}', 'review')
    os.makedirs(rev, exist_ok=True)
    with open(os.path.join(rev, 'numeric_audit.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


if __name__ == '__main__':
    results = {k: audit_leg(k, v) for k, v in LEGS.items()}
    print('\n===== SUMMARY =====')
    for k, v in results.items():
        print(f"{k}: {v['verdict']}")
