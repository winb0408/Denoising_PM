#!/usr/bin/env python
"""E2-nf 全栈 tiled 推理(VALID 原版 ckpt)+ 数值定性代理审计 + 统一可视化。

推理: 64×128×128 无重叠 tiling, mean/std 归一化输入, 输出去归一化, 存 uint16。
审计: 与 SN2N legs 同款指标(std 比/Pearson/局部方差相关/饱和/SBR)。
可视化: qmask.render_qualitative 同款裁剪与 magma 渲染, 3 列(input|pred|target)。
"""
import json
import sys

import numpy as np
import tifffile
import torch

sys.path.insert(0, '/data2/wjb/Denoising_PM/ACES')
from qmask.visualize import _robust_lims  # 渲染约定唯一来源

CKPT_DIR = ('/data2/wjb/Denoising_PM/Methods/VALID-v1.0/FDU-donglab-VALID-34f5637/'
            'checkpoint/train_neurofinder_00_00202609041902')
STACK = '/data2/wjb/Denoising_PM/ACES/data/neurofinder.00.00/train/neurofinder_00_00_stack.tif'
TARGET = '/data2/wjb/Denoising_PM/ACES/data/neurofinder.00.00/proxy/neurofinder_00_00_temporal_mean.tif'
ROI = '/data2/wjb/Denoising_PM/ACES/data/neurofinder.00.00/proxy/neurofinder_00_00_roi_mask.tif'
OUTDIR = '/data3/wjb/Denoising_PM/Results/valid_original_neurofinder'
PZ, PW, PH = 64, 128, 128

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device('cuda:0')
sys.path.insert(0, '/data2/wjb/Denoising_PM/Methods/VALID-v1.0/FDU-donglab-VALID-34f5637')
from models.network import Network_CNR

cfg = json.load(open(CKPT_DIR + '/config.json'))
model = Network_CNR(1, 1, f_maps=int(cfg['base_features']), n_groups=int(cfg['n_groups'])).to(device)
ckpt = torch.load(CKPT_DIR + '/train_neurofinder_00_00_Epoch_100.pth', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['state_dict'])
model.eval()

stack = tifffile.imread(STACK).astype(np.float32)
mean_v, std_v = float(stack.mean()), float(stack.std())
z, y, x = stack.shape
nz, ny, nx = z // PZ, y // PW, x // PH
print(f'stack {stack.shape}, tiles {nz}x{ny}x{nx}')

pred = np.zeros((nz * PZ, ny * PW, nx * PH), dtype=np.float32)
coords = [(iz, iy, ix) for iz in range(nz) for iy in range(ny) for ix in range(nx)]
with torch.no_grad():
    for start in range(0, len(coords), 8):
        sub = coords[start:start + 8]
        arr = np.stack([(stack[c[0]*PZ:(c[0]+1)*PZ, c[1]*PW:(c[1]+1)*PW, c[2]*PH:(c[2]+1)*PH] - mean_v) / std_v
                        for c in sub]).astype(np.float32)
        t = torch.from_numpy(arr).unsqueeze(1).to(device)
        o = model(t).squeeze(1).detach().cpu().numpy().astype(np.float32) * std_v + mean_v
        for k, c in enumerate(sub):
            pred[c[0]*PZ:(c[0]+1)*PZ, c[1]*PW:(c[1]+1)*PW, c[2]*PH:(c[2]+1)*PH] = o[k]
        if (start // 8) % 200 == 0:
            print(f'batch {start // 64}/{len(coords) // 8}')
tifffile.imwrite(OUTDIR + '/predictions_full.tif', pred.astype(np.uint16))
print('saved predictions_full.tif')

# ---- 数值审计(与 SN2N legs 同款)----
target = tifffile.imread(TARGET).astype(np.float32)[:ny*PW, :nx*PH]
roi = tifffile.imread(ROI).astype(bool)[:ny*PW, :nx*PH]
z2 = nz * PZ
tgt3 = np.broadcast_to(target, (z2, ny*PW, nx*PH)).astype(np.float32, copy=False)
roi3 = np.broadcast_to(roi, (z2, ny*PW, nx*PH))
bg3 = ~roi3


def stats(m):
    xx = pred[m].astype(np.float64); yy = tgt3[m].astype(np.float64)
    return float(np.corrcoef(xx, yy)[0, 1]), float(xx.std() / max(yy.std(), 1e-9))


cf, sf = stats(roi3); cb, sb = stats(bg3); ca, sa = stats(np.ones_like(roi3))
lv = []
for z0 in range(0, z2, 256):
    z1 = min(z0 + 256, z2)
    def bstd(a2):
        H, W = (a2.shape[0]//8)*8, (a2.shape[1]//8)*8
        return a2[:H, :W].reshape(H//8, 8, W//8, 8).std(axis=(1, 3))
    lv.append(float(np.corrcoef(bstd(pred[z0:z1].mean(0)).ravel(),
                                bstd(tgt3[z0:z1].mean(0)).ravel())[0, 1]))
sbr_t = float(tgt3[roi3].mean() / max(tgt3[bg3].mean(), 1e-9))
sbr_p = float(pred[roi3].mean() / max(pred[bg3].mean(), 1e-9))
audit = {'fg_pearson': cf, 'fg_std_ratio': sf, 'bg_pearson': cb, 'bg_std_ratio': sb,
         'all_pearson': ca, 'all_std_ratio': sa, 'local_var_corr': float(np.mean(lv)),
         'sbr_target': sbr_t, 'sbr_pred': sbr_p, 'sbr_kept': sbr_p / max(sbr_t, 1e-9)}
flags = []
if sf < 0.4: flags.append('COLLAPSE')
if cf < 0.3: flags.append('STRUCT-LOSS')
if audit['local_var_corr'] < 0: flags.append('TEXTURE-INVERT')
audit['verdict'] = 'PASS' if not flags else ';'.join(flags)
print(json.dumps(audit, indent=1))
json.dump(audit, open(OUTDIR + '/numeric_audit.json', 'w'), indent=1)

# ---- 统一可视化(qmask 同款裁剪)----
z0 = stack.shape[0] // 4
y0, x0 = stack.shape[1] // 4, stack.shape[2] // 4
mid = 16
panels = [
    ('input frame %d' % (z0 + mid), stack[z0 + mid, y0:y0+256, x0:x0+256]),
    ('VALID-original', pred[z0 + mid, y0:y0+256, x0:x0+256]),
    ('target/temporal-mean', target[y0:y0+256, x0:x0+256]),
]
fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
for col, (title, img) in enumerate(panels):
    vmin, vmax = _robust_lims(img)
    axes[0][col].imshow(img, cmap='magma', vmin=vmin, vmax=vmax)
    axes[0][col].set_title(f'{title}\n[{vmin:.0f},{vmax:.0f}]', fontsize=8)
    axes[0][col].axis('off')
fig.tight_layout()
fig.savefig(OUTDIR + '/figures/qualitative_comparison_ext.png', dpi=100, bbox_inches='tight')
print('saved figures/qualitative_comparison_ext.png')
