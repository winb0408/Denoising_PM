#!/usr/bin/env python
"""E1-nf-2D 统一尺 eval（与 E1-nf-3D step3c 完全同协议）+ 数值定性代理审计。
协议: 18 fg-rich patches (PZ,PW,PH=64,128,128, stride16), per-patch p0.1–p99.9
data_range, 全局 lstsq 线性对齐, 2D temporal-mean/ROI 沿 z 广播。
"""
import glob
import json
import os

import numpy as np
import tifffile
from scipy import ndimage

from aces.evaluate_stage2 import _region_metrics, _mask_patch

OUT = '/data3/wjb/Denoising_PM/Results/sn2n_original_neurofinder_2d'
PROXY = '/data2/wjb/Denoising_PM/ACES/data/neurofinder.00.00/proxy'

pred = tifffile.imread(glob.glob(OUT + '/predictions/*.tif')[0]).astype(np.float32)
target = tifffile.imread(PROXY + '/neurofinder_00_00_temporal_mean.tif').astype(np.float32)
roi = tifffile.imread(PROXY + '/neurofinder_00_00_roi_mask.tif').astype(bool)
assert target.ndim == 2 and roi.ndim == 2
z, y, x = pred.shape
target3 = np.broadcast_to(target, (z, y, x)).astype(np.float32, copy=False)
roi3 = np.broadcast_to(roi, (z, y, x)).astype(bool, copy=False)

# 全局线性对齐（自监督输出无绝对尺度）
A = np.stack([pred.ravel(), np.ones(pred.size)], axis=1)
coef, *_ = np.linalg.lstsq(A, target3.ravel(), rcond=None)
a, b = float(coef[0]), float(coef[1])
print(f'global lstsq: pred_aligned = {a:.4f}*pred + {b:.2f}')
aligned = a * pred + b

PZ, PW, PH = 64, 128, 128
cands = [(int(_mask_patch(roi, z0, y0, x0, (PZ, PW, PH)).sum()), z0, y0, x0)
         for z0 in range(0, z - PZ + 1, 16)
         for y0 in range(0, y - PW + 1, 16)
         for x0 in range(0, x - PH + 1, 16)]
cands.sort(reverse=True)
selected = cands[:18]
print(f'18 fg-rich patches selected (fg frac {selected[0][0]/ (PZ*PW*PH):.3f} .. {selected[-1][0]/(PZ*PW*PH):.3f})')

scores = {'all': [], 'foreground': [], 'background': []}
for _, z0, y0, x0 in selected:
    pp = aligned[z0:z0+PZ, y0:y0+PW, x0:x0+PH]
    tp = target3[z0:z0+PZ, y0:y0+PW, x0:x0+PH]
    mp = roi3[z0:z0+PZ, y0:y0+PW, x0:x0+PH]
    for name, m in (('all', np.ones_like(mp)), ('foreground', mp), ('background', ~mp)):
        scores[name].append(_region_metrics(pp, tp, m)['psnr_db'])

res = {k: float(np.mean(v)) for k, v in scores.items()}
print('patch eval (18 patches, mean PSNR dB):', res)

# ---- 数值定性代理审计（同 audit_e1_numeric.py 逻辑）----
def block_std(a2d, block=8):
    H, W = (a2d.shape[0]//block)*block, (a2d.shape[1]//block)*block
    return a2d[:H, :W].reshape(H//block, block, W//block, block).std(axis=(1, 3))

lo, hi = float(pred.min()), float(pred.max())
sat_lo = float((pred <= lo + 0.5).mean())
stats = {r: dict(n=0, sx=0.0, sy=0.0, sxx=0.0, syy=0.0, sxy=0.0) for r in ('all', 'fg', 'bg')}
lv_p, lv_t, dead, tot = [], [], 0, 0
bg3 = ~roi3
for z0 in range(0, z, 64):
    z1 = min(z0 + 64, z)
    pa, ta = aligned[z0:z1], target3[z0:z1]
    for r, m in (('all', None), ('fg', roi3[z0:z1]), ('bg', bg3[z0:z1])):
        xx = (pa if m is None else pa[m]).ravel().astype(np.float64)
        yy = (ta if m is None else ta[m]).ravel().astype(np.float64)
        s = stats[r]
        s['n'] += xx.size; s['sx'] += xx.sum(); s['sy'] += yy.sum()
        s['sxx'] += (xx*xx).sum(); s['syy'] += (yy*yy).sum(); s['sxy'] += (xx*yy).sum()
    lp, lt = block_std(pa.mean(axis=0)), block_std(ta.mean(axis=0))
    lv_p.append(lp); lv_t.append(lt)
    dead += int((lp < 1e-6).sum()); tot += lp.size

fin = {}
for r, s in stats.items():
    n = max(s['n'], 1)
    mx, my = s['sx']/n, s['sy']/n
    vx, vy = max(s['sxx']/n - mx*mx, 0.0), max(s['syy']/n - my*my, 0.0)
    cov = s['sxy']/n - mx*my
    fin[r] = {'std_ratio': np.sqrt(vx)/max(np.sqrt(vy), 1e-9),
              'pearson': cov/max(np.sqrt(vx*vy), 1e-12)}
lv_corr = float(np.corrcoef(np.concatenate([v.ravel() for v in lv_p]),
                            np.concatenate([v.ravel() for v in lv_t]))[0, 1])
mean_t_fg = float(target3[roi3].astype(np.float64).mean())
mean_t_bg = float(target3[bg3].astype(np.float64).mean())
sbr_t = mean_t_fg/max(mean_t_bg, 1e-9)
sbr_p = (fin['fg'] and (aligned[roi3].mean()/max(aligned[bg3].mean(), 1e-9)))

audit = {
    'raw_pred_minmax': [lo, hi], 'sat_low_frac': sat_lo,
    'dead_block_frac': dead/max(tot, 1), 'local_var_corr': lv_corr,
    'std_ratio': {r: float(fin[r]['std_ratio']) for r in fin},
    'pearson': {r: float(fin[r]['pearson']) for r in fin},
    'sbr_target': sbr_t, 'sbr_pred_aligned': float(sbr_p),
    'sbr_kept_frac': float(sbr_p/max(sbr_t, 1e-9)),
}
flags = []
if audit['std_ratio']['fg'] < 0.4: flags.append('COLLAPSE')
if audit['pearson']['fg'] < 0.3: flags.append('STRUCT-LOSS')
if audit['sat_low_frac'] > 0.05: flags.append('SATURATION')
if audit['dead_block_frac'] > 0.10: flags.append('DEAD-BLOCKS')
if audit['local_var_corr'] < 0: flags.append('TEXTURE-INVERT')
audit['verdict'] = 'PASS' if not flags else ';'.join(flags)
print('audit:', json.dumps(audit, indent=1))
print('verdict:', audit['verdict'])

out = {'scale_fit': {'a': a, 'b': b}, 'patch_eval_db': res, 'audit': audit}
with open(OUT + '/patch_eval_metrics.json', 'w') as f:
    json.dump(out, f, indent=2)
print('saved', OUT + '/patch_eval_metrics.json')
