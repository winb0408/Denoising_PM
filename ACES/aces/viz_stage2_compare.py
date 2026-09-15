"""Stage 2 五档综合可视化 + 信背比(SBR)分析。

目的（对应技术方案 §8 成功标准）：不只看前景 PSNR，还要回答
"创新的各向异性采样档是否提升信背比、抑制背景干扰"，并给出可视化对比。

对每个采样档：加载 best checkpoint，在同一批前景丰富的全分辨率 patch 上推理，
把输出重采样回全分辨率参考网格（下采样档 valid/sn2n_xy/iso3d 的网络输出分辨率低于输入，
用最近邻放大以便与 mean9 GT 和其它档在同一网格上比较，并在图中标注），
计算：前景/背景 PSNR、SSIM_fg、信背比 SBR=均值(前景)/均值(背景)、
背景残余噪声 std_bg、背景抑制比(相对含噪输入)。并输出 XY / XZ 切片对比图。

用法：
  CUDA_VISIBLE_DEVICES=<n> PYTHONPATH=ACES python -m aces.viz_stage2_compare \
    --runs-root ACES/runs --tag 40ep_seed3407 --out ACES/runs/stage02_compare_40ep \
    --n-patches 3 --ckpt best
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim

from .io_utils import load_config, write_json
from .paths import MEAN9_PROXY, VALID_ROOT
from .profile_anisotropy import build_region_masks

MODES = ["valid", "sn2n_xy", "iso3d", "empirical_block", "ellipsoid"]
# 端点基线 vs 创新档，用于结论归因（方案 §1、§8）。
BASELINE_MODES = {"valid", "sn2n_xy"}
NOVEL_MODES = {"empirical_block", "ellipsoid"}

def _upsample_to(arr: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """把网络输出（可能因下采样档而变小）用最近邻放大到参考网格 shape。"""
    if arr.shape == shape:
        return arr
    factors = [s / a for s, a in zip(shape, arr.shape)]
    out = ndimage.zoom(arr, factors, order=0)
    # zoom 可能有 ±1 偏差，裁/补到精确 shape
    out = out[: shape[0], : shape[1], : shape[2]]
    if out.shape != shape:
        pad = [(0, s - o) for s, o in zip(shape, out.shape)]
        out = np.pad(out, pad, mode="edge")
    return out


def _load_model(checkpoint_path: Path, train_cfg: dict, device: torch.device):
    import sys

    sys.path.insert(0, str(VALID_ROOT))
    from models.network import Network_CNR

    model = Network_CNR(
        in_channels=1,
        out_channels=1,
        f_maps=int(train_cfg["base_features"]),
        n_groups=int(train_cfg["n_groups"]),
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _metrics(pred: np.ndarray, ref: np.ndarray, masks: dict, noisy: np.ndarray) -> dict:
    """前景/背景分层指标 + 信背比(SBR) + 背景抑制。"""
    # 边界 patch 处 ref/mask 可能被体积边界截短，而 pred 保持完整 patch 尺寸，
    # 会导致形状不一致（SSIM 报错）。统一裁到三者公共最小形状后再计算。
    cz = min(pred.shape[0], ref.shape[0], noisy.shape[0], masks["foreground"].shape[0])
    cy = min(pred.shape[1], ref.shape[1], noisy.shape[1], masks["foreground"].shape[1])
    cx = min(pred.shape[2], ref.shape[2], noisy.shape[2], masks["foreground"].shape[2])
    pred = pred[:cz, :cy, :cx]
    ref = ref[:cz, :cy, :cx]
    noisy = noisy[:cz, :cy, :cx]
    masks = {k: v[:cz, :cy, :cx] for k, v in masks.items()}
    fg = masks["foreground"]
    bg = masks["background"]

    def psnr(region):
        # 口径与验收 evaluate_stage2._region_metrics 完全一致：
        # data_range 取"区域内" ref 的 99.9/0.1 分位数之差（而非全图），
        # 否则前景(高信号区)会因借用全图 data_range 被严重低估。
        if not region.any():
            return float("nan")
        ref_r = ref[region].astype(np.float64)
        d = pred[region].astype(np.float64) - ref_r
        mse = float(np.mean(d * d))
        dr = max(float(np.percentile(ref_r, 99.9) - np.percentile(ref_r, 0.1)), 1e-6)
        return float("inf") if mse == 0 else 10.0 * np.log10(dr * dr / mse)

    # SSIM 需在归一化后整体计算，前景版本用 mask 加权近似（先算全图 map 再取前景均值）
    rmin, rmax = float(ref.min()), float(ref.max())
    scale = max(rmax - rmin, 1e-6)
    p_n = np.clip((pred - rmin) / scale, 0, 1)
    r_n = np.clip((ref - rmin) / scale, 0, 1)
    _, ssim_map = ssim(r_n, p_n, data_range=1.0, full=True)
    ssim_fg = float(ssim_map[fg].mean()) if fg.any() else float("nan")
    ssim_all = float(ssim_map.mean())

    # 信背比：前景信号均值 / 背景均值；背景残余噪声：背景 std
    fg_mean = float(pred[fg].mean()) if fg.any() else float("nan")
    bg_mean = float(pred[bg].mean()) if bg.any() else float("nan")
    sbr_pred = fg_mean / bg_mean if bg_mean not in (0.0, float("nan")) else float("nan")
    fg_mean_in = float(noisy[fg].mean()) if fg.any() else float("nan")
    bg_mean_in = float(noisy[bg].mean()) if bg.any() else float("nan")
    sbr_in = fg_mean_in / bg_mean_in if bg_mean_in else float("nan")
    sbr_gt_fg = float(ref[fg].mean()) if fg.any() else float("nan")
    sbr_gt_bg = float(ref[bg].mean()) if bg.any() else float("nan")
    sbr_gt = sbr_gt_fg / sbr_gt_bg if sbr_gt_bg else float("nan")

    std_bg_pred = float(pred[bg].std()) if bg.any() else float("nan")
    std_bg_in = float(noisy[bg].std()) if bg.any() else float("nan")
    bg_suppression = std_bg_in / std_bg_pred if std_bg_pred else float("nan")
    return {
        "psnr_fg": psnr(fg),
        "psnr_bg": psnr(bg),
        "psnr_all": psnr(np.ones_like(fg)),
        "ssim_fg": ssim_fg,
        "ssim_all": ssim_all,
        "sbr_pred": sbr_pred,
        "sbr_input": sbr_in,
        "sbr_gt": sbr_gt,
        "std_bg_pred": std_bg_pred,
        "std_bg_input": std_bg_in,
        "bg_noise_suppression": bg_suppression,
    }

def run(runs_root: Path, tag: str, out_dir: Path, n_patches: int, ckpt_kind: str,
        run_map: dict | None = None) -> dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    # 档位与 checkpoint 路径解析：
    # - 默认（run_map=None）沿用五档写死方案 stage02_{mode}_{tag}/{mode}/checkpoints/{mode}_{ckpt}.pth。
    # - run_map={label: (run_subdir, sampling_mode)} 时支持任意档位对比（如叠加背景正则的 ellipsoid_bgreg 档），
    #   run_subdir 为 runs_root 下的目录名，sampling_mode 为其内部子目录 / checkpoint 前缀。
    if run_map is None:
        modes = list(MODES)
        ckpt_paths = {
            m: runs_root / f"stage02_{m}_{tag}" / m / "checkpoints" / f"{m}_{ckpt_kind}.pth"
            for m in modes
        }
    else:
        modes = list(run_map.keys())
        ckpt_paths = {
            label: runs_root / sub / smode / "checkpoints" / f"{smode}_{ckpt_kind}.pth"
            for label, (sub, smode) in run_map.items()
        }

    # 用 ellipsoid 档的 config 读取 patch 几何与 mean9 路径（各档 training/eval 一致）
    ref_cfg_path = runs_root / f"stage02_ellipsoid_{tag}" / "ellipsoid" / "config.json"
    cfg = json.loads(ref_cfg_path.read_text())
    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]
    mean9 = tifffile.imread(eval_cfg.get("mean9_proxy", str(MEAN9_PROXY))).astype(np.float32)
    masks_full = build_region_masks(mean9, percentile=float(eval_cfg.get("foreground_percentile", 75.0)))

    import sys

    sys.path.insert(0, str(VALID_ROOT))
    from datasets.dataset_fs import ReadDatasets

    # 关键：必须与训练期评估 (evaluate_stage2 复用的训练 dataset) 完全同口径。
    # 训练期用 mode="train" + patch_num=train_cfg["patch_num"](=-1，枚举全部重叠 patch)，
    # 保持 z_patch=64；若误用 mode="test"，dataset 会把 z_patch 重算成整深 137、
    # 只剩 18 个大块，patch 网格与选择全不同，PSNR 会系统性偏低约 8 dB。
    dataset = ReadDatasets(
        dataPath=train_cfg["train_folder"], mode="train", dataType="3D", dataExtension="tif",
        z_patch=int(train_cfg["z_patch"]), w_patch=int(train_cfg["w_patch"]), h_patch=int(train_cfg["h_patch"]),
        z_overlap=float(train_cfg["z_overlap"]), w_overlap=float(train_cfg["w_overlap"]), h_overlap=float(train_cfg["h_overlap"]),
        patch_num=int(train_cfg["patch_num"]), dataNum=int(train_cfg.get("train_frame_num", 10000)),
    )
    # 选前景最丰富的 n_patches 个 patch（与评估一致的口径）
    cand = []
    for i in range(len(dataset)):
        _, img_idx, z_idx, y_idx, x_idx = dataset[i]
        if int(img_idx) != 0:
            continue
        z0, y0, x0 = int(z_idx), int(y_idx), int(x_idx)
        z1 = min(z0 + dataset.z_patch, masks_full["foreground"].shape[0])
        y1 = min(y0 + dataset.w_patch, masks_full["foreground"].shape[1])
        x1 = min(x0 + dataset.h_patch, masks_full["foreground"].shape[2])
        cand.append((int(masks_full["foreground"][z0:z1, y0:y1, x0:x1].sum()), i, (z0, y0, x0)))
    cand.sort(reverse=True)
    chosen = cand[:n_patches]

    # 预加载各档模型
    models = {}
    for m in modes:
        models[m] = _load_model(ckpt_paths[m], train_cfg, device)

    all_rows: list[dict] = []
    for rank, (_, item_index, (z0, y0, x0)) in enumerate(chosen):
        patch, img_idx, _, _, _ = dataset[item_index]
        input_np = patch.squeeze(0).numpy() if patch.dim() == 4 else patch.numpy()
        shp = input_np.shape
        ref = mean9[z0:z0 + shp[0], y0:y0 + shp[1], x0:x0 + shp[2]]
        pmask = {k: v[z0:z0 + shp[0], y0:y0 + shp[1], x0:x0 + shp[2]] for k, v in masks_full.items()}
        # 含噪输入反归一化到与 mean9 同量纲
        noisy = input_np * float(dataset.image_stds[0]) + float(dataset.image_means[0])

        preds = {}
        with torch.no_grad():
            for m in modes:
                it = torch.from_numpy(np.ascontiguousarray(input_np)).unsqueeze(0).unsqueeze(0).to(device)
                o = models[m](it).squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
                o = o * float(dataset.image_stds[0]) + float(dataset.image_means[0])
                preds[m] = _upsample_to(o, shp)
                met = _metrics(preds[m], ref, pmask, noisy)
                met.update({"mode": m, "patch_index": item_index, "patch_rank": rank})
                all_rows.append(met)

        _plot_patch(out_dir / "figures" / f"patch{rank}_idx{item_index}.png",
                    noisy, ref, preds, pmask, item_index, (z0, y0, x0), modes)

    # 聚合：各档在所选 patch 上的均值
    agg: dict[str, dict] = {}
    keys = ["psnr_fg", "psnr_bg", "ssim_fg", "sbr_pred", "std_bg_pred", "bg_noise_suppression"]
    for m in modes:
        rows = [r for r in all_rows if r["mode"] == m]
        agg[m] = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    # 参考基准：输入 SBR / GT SBR（与档无关，取任意 patch 行的均值）
    ref_sbr = {
        "sbr_input": float(np.nanmean([r["sbr_input"] for r in all_rows])),
        "sbr_gt": float(np.nanmean([r["sbr_gt"] for r in all_rows])),
        "std_bg_input": float(np.nanmean([r["std_bg_input"] for r in all_rows])),
    }
    result = {"tag": tag, "n_patches": n_patches, "ckpt": ckpt_kind, "modes": modes,
              "per_mode": agg, "reference": ref_sbr, "per_patch_rows": all_rows}
    write_json(out_dir / "sbr_metrics.json", result)
    _write_md(out_dir / "sbr_report.md", agg, ref_sbr, tag, n_patches, ckpt_kind, modes)
    return result

def _plot_patch(path: Path, noisy, ref, preds, masks, item_index, origin, modes):
    """一张图：上排 XY 中层切片，下排 XZ 侧视切片；列= 含噪输入 / 各档 / mean9 GT。"""
    zc = noisy.shape[0] // 2
    yc = noisy.shape[1] // 2
    cols = ["input(noisy)"] + modes + ["mean9 GT"]
    xy_imgs = [noisy[zc]] + [preds[m][zc] for m in modes] + [ref[zc]]
    xz_imgs = [noisy[:, yc, :]] + [preds[m][:, yc, :] for m in modes] + [ref[:, yc, :]]
    # 统一显示范围，用 GT 前景稳健分位
    vmin = float(np.percentile(ref, 1))
    vmax = float(np.percentile(ref, 99.5))
    n = len(cols)
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.4))
    for j in range(n):
        axes[0, j].imshow(xy_imgs[j], cmap="magma", vmin=vmin, vmax=vmax)
        axes[0, j].set_title(cols[j], fontsize=8)
        axes[0, j].axis("off")
        axes[1, j].imshow(xz_imgs[j], cmap="magma", vmin=vmin, vmax=vmax, aspect="auto")
        axes[1, j].axis("off")
    axes[0, 0].set_ylabel("XY (z-mid)", fontsize=8)
    axes[1, 0].set_ylabel("XZ (y-mid)", fontsize=8)
    fig.suptitle(f"patch idx={item_index} origin(z,y,x)={origin}   top:XY  bottom:XZ(axial)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _write_md(path: Path, agg, ref_sbr, tag, n_patches, ckpt_kind, modes):
    lines = [
        "# ACES Stage 2 五档综合分析（PSNR + SSIM + 信背比 SBR + 背景抑制）",
        "",
        f"- tag: `{tag}`  checkpoint: `{ckpt_kind}`  评估 patch 数: {n_patches}（前景最丰富）",
        f"- 含噪输入 SBR = {ref_sbr['sbr_input']:.4f}，mean9 GT SBR = {ref_sbr['sbr_gt']:.4f}，输入背景 std = {ref_sbr['std_bg_input']:.1f}",
        "- SBR=前景均值/背景均值（越高=背景相对被压得越低，信号越突出）。",
        "- bg_noise_suppression=输入背景std/输出背景std（>1=背景噪声被抑制，越大越干净）。",
        "- baseline 端点: valid / sn2n_xy；创新档: empirical_block / ellipsoid；对照: iso3d。",
        "",
        "| mode | PSNR_fg | PSNR_bg | SSIM_fg | SBR_pred | std_bg_pred | bg_noise_suppression |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in modes:
        a = agg[m]
        lines.append(
            f"| {m} | {a['psnr_fg']:.3f} | {a['psnr_bg']:.3f} | {a['ssim_fg']:.4f} | "
            f"{a['sbr_pred']:.4f} | {a['std_bg_pred']:.1f} | {a['bg_noise_suppression']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ACES Stage2 五档可视化 + SBR 综合分析")
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--tag", required=True, help="如 40ep_seed3407")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-patches", type=int, default=3)
    ap.add_argument("--ckpt", default="best", choices=["best", "final"])
    ap.add_argument(
        "--run-map", default=None,
        help=("可选。JSON 字符串，形如 "
              "'{\"ellipsoid\":[\"stage02_ellipsoid_40ep_seed3407\",\"ellipsoid\"],"
              "\"ellipsoid_bgreg005\":[\"stage02_ellipsoid_bgreg005_40ep_seed3407\",\"ellipsoid\"]}'。"
              "键为列标签，值为 [run目录名, 采样模式/子目录]。给定后忽略默认五档，按此表对比任意档位。"),
    )
    args = ap.parse_args(argv)
    run_map = None
    if args.run_map:
        raw = json.loads(args.run_map)
        run_map = {k: (v[0], v[1]) for k, v in raw.items()}
    res = run(Path(args.runs_root), args.tag, Path(args.out), args.n_patches, args.ckpt, run_map)
    print(json.dumps(res["per_mode"], indent=2))
    print("reference:", json.dumps(res["reference"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())