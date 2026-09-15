#!/usr/bin/env python
"""外部基线可视化统一方案（2026-09-04 修改小结 #1）。

用户要求：外部仓库对比的可视化必须与五档归因实验（qmask.visualize.render_qualitative）
保持一致。本脚本直接 import qmask.visualize 的渲染约定, 不另起炉灶：
  - colormap = magma
  - 每面板限幅 = _robust_lims (p0.5~p99.5)
  - 裁剪 = run_dir 布局同款: z 带 = [shape//4, +32), y/x = [shape//4, +256), mid = 带内 16
  - 列 = input | <method> pred | target/mean9（外部单模型无 best/final 之分, 3 列）
  - 每面板标题标注实际 [vmin, vmax], 便于跨实验对照

用法: PYTHONPATH=ACES python ACES/logs/render_e1_viz.py [--leg nf|3p|all]
输出: <run_dir>/figures/qualitative_comparison_ext.png + review 切块（qmask.preview 同款）
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tifffile

sys.path.insert(0, '/data2/wjb/Denoising_PM/ACES')
from qmask.visualize import _robust_lims  # 渲染约定唯一来源, 禁止在本脚本重定义

OUT_ROOT = '/data3/wjb/Denoising_PM/Results'
MEAN9_PROXY = ('/data2/wjb/Denoising_PM/Methods/FAST-main/reproduction/'
               'valid_comparison/experiments/valid_paper_3p_neuron/'
               'original_workflow_paper_params_repeat01_seed3407/data/gt_proxy/'
               '3P_neuron_mean9_gt_proxy_275z.tif')

# 每个 leg: run 目录 + 方法名 + pred 路径模式; noisy/target 沿用 run 布局
LEGS = {
    'nf': {
        'run_dir': OUT_ROOT + '/sn2n_original_neurofinder',
        'method': 'SN2N-3D-p2p1',
        'pred_glob': '/predictions/*.tif',
        'a': 7.5770, 'b': 24.67,  # 全局线性对齐（自监督输出无绝对尺度, 已文档化）
    },
    '3p': {
        'run_dir': OUT_ROOT + '/sn2n_original_3p',
        'method': 'SN2N-3D-p2p1',
        'pred_glob': '/predictions/*.tif',
        'a': 131.198668662158, 'b': 871.1020115804515,
    },
    # 重跑 leg（2026-09-04 小结 #1/#2）
    'nf2d': {
        'run_dir': OUT_ROOT + '/sn2n_original_neurofinder_2d',
        'method': 'SN2N-2D-p2p1',
        'pred_glob': '/predictions/*.tif',
        'a': 2.9023, 'b': 86.06,
        'noisy': '/data2/wjb/Denoising_PM/ACES/data/neurofinder.00.00/train/neurofinder_00_00_stack.tif',
    },
    '3p_p2p2': {
        'run_dir': OUT_ROOT + '/sn2n_original_3p_p2p2',
        'method': 'SN2N-3D-p2p2',
        'pred_glob': '/predictions/*.tif',
        'a': 172.8727, 'b': -8.19,
        'noisy': OUT_ROOT + '/sn2n_original_3p_p2p2/raw_data_8bit/3P_neuron_repeat01_275z_8bit.tif',
    },
}


def render_leg(leg, cfg):
    import glob
    run_dir = cfg['run_dir']
    pred_path = sorted(glob.glob(run_dir + cfg['pred_glob']))
    assert pred_path, f'{leg}: no prediction under {run_dir}'
    pred = tifffile.imread(pred_path[-1]).astype(np.float32)
    if 'noisy' in cfg:
        noisy = tifffile.imread(cfg['noisy']).astype(np.float32)
    else:
        noisy_dir = run_dir + '/raw_data_8bit'
        noisy_name = [f for f in os.listdir(noisy_dir) if f.endswith('.tif')][0]
        noisy = tifffile.imread(os.path.join(noisy_dir, noisy_name)).astype(np.float32)
    if leg in ('nf', 'nf2d'):
        target = tifffile.imread('/data2/wjb/Denoising_PM/ACES/data/neurofinder.00.00/proxy/'
                                 'neurofinder_00_00_temporal_mean.tif').astype(np.float32)
    else:
        target = tifffile.imread(MEAN9_PROXY).astype(np.float32)

    # 同 qmask.render_qualitative 的裁剪几何
    z0 = noisy.shape[0] // 4
    y0, x0 = noisy.shape[1] // 4, noisy.shape[2] // 4
    noisy_c = noisy[z0:z0 + 32, y0:y0 + 256, x0:x0 + 256]
    pred_c = cfg['a'] * pred + cfg['b']
    pred_c = pred_c[z0:z0 + 32, y0:y0 + 256, x0:x0 + 256]
    mid = 16
    target_c = target[y0:y0 + 256, x0:x0 + 256] if target.ndim == 2 else \
        target[z0 + mid, y0:y0 + 256, x0:x0 + 256]

    panels = [
        ('input frame %d' % mid, noisy_c[mid]),
        (cfg['method'], pred_c[mid]),
        ('target/mean9', target_c),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
    for col, (title, img) in enumerate(panels):
        vmin, vmax = _robust_lims(img)
        axes[0][col].imshow(img, cmap='magma', vmin=vmin, vmax=vmax)
        axes[0][col].set_title(f'{title}\n[{vmin:.0f},{vmax:.0f}]', fontsize=8)
        axes[0][col].axis('off')
    fig.tight_layout()
    out = os.path.join(run_dir, 'figures', 'qualitative_comparison_ext.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'[{leg}] saved {out} ({os.path.getsize(out)/1e6:.2f} MB)')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--leg', default='all', choices=['nf', '3p', 'nf2d', '3p_p2p2', 'all'])
    a = ap.parse_args()
    legs = LEGS if a.leg == 'all' else {a.leg: LEGS[a.leg]}
    for k, v in legs.items():
        render_leg(k, v)
