"""Paper-style 固化可视化流程（唯一入口；新实验一律复用本模块，禁止重写）。

约定固化自 Results/neurofinder_00_00_multimethod_comparison_20260822/scripts：
- 伪彩 colormap=turbo；overview 共享窗 (1.0, 99.85)；ROI zoom 每块 (1.0, 99.9)
- 方法顺序/标签/颜色固定（METHOD_ORDER / METHOD_LABELS / METHOD_COLORS）
- 指标条形图：3-seed mean ± 半程误差棒；附 paper-style 汇总表
- 任何 >1200px 大图落盘后必须立即生成 review 切块，定性检查只读切块
  （CLAUDE.md 规则，防多模态 API 卡死）

入口：
  python -m qmask.paperviz --run-dir  <stage2_run_dir>                 # 单 run 多臂定性
  python -m qmask.paperviz --matrix-root <dir> --seeds 3407,3408,3409  # 重复矩阵定性+定量
产出（--out 目录，默认 <matrix_root>/paperviz）：
  pseudocolor_overview.png   turbo 伪彩 MIP（ROI 框选）
  roi_zooms.png              ROI 放大网格（target 红轮廓）
  patch_rows.png             评分 ruler top-3 fg patch：noisy|各臂|target
  metric_bars.png            all/fg/bg PSNR 误差棒
  summary_table.png          汇总表
  review/                    所有图的 ≤1200px 切块（只读这些）
  index.html                 汇总页
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from matplotlib.patches import Rectangle

from aces.paths import VALID_ROOT
from .preview import _rgb_copy, _save_bounded_png

# ---------------------------------------------------------------- 固定约定
METHOD_ORDER = ["noisy", "valid", "sn2n_xy", "iso3d", "qmask", "target"]


def _ordered_methods(keys) -> list[str]:
    """已知臂按 METHOD_ORDER 排，自定义臂（--arm 跨 run 对比）按插入序排在后。"""
    known = [m for m in METHOD_ORDER if m in keys]
    extra = [m for m in keys if m not in METHOD_ORDER]
    return known + extra


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _color(method: str):
    return METHOD_COLORS.get(method, None)
METHOD_LABELS = {
    "noisy": "Noisy",
    "valid": "VALID",
    "sn2n_xy": "SN2N-xy",
    "iso3d": "Iso-3D",
    "qmask": "QMask",
    "target": "Target (proxy)",
}
METHOD_COLORS = {
    "noisy": "#9e9e9e",
    "valid": "#4c72b0",
    "sn2n_xy": "#dd8452",
    "iso3d": "#55a868",
    "qmask": "#c44e52",
    "target": "#3b3b3b",
}
OVERVIEW_PCTL = (1.0, 99.85)  # overview 共享窗（与参考代码一致）
ZOOM_PCTL = (1.0, 99.9)  # ROI zoom 每块窗
CMAP = "turbo"
ROI_SIZE = 84
ROI_LIMIT = 4

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def _normalize(image: np.ndarray, lo: float, hi: float) -> np.ndarray:
    denom = max(hi - lo, 1e-6)
    return np.clip((image.astype(np.float32) - lo) / denom, 0.0, 1.0)


# ---------------------------------------------------------------- review 切块
def make_figure_tiles(fig_path: str | Path, tiles_dir: str | Path, row_height: int = 280,
                      max_bytes: int = 3_500_000) -> list[dict[str, Any]]:
    """对任意落盘大图生成 ≤1200px 行切块（CLAUDE.md 强制规则），返回 manifest 行。"""
    fig_path, tiles_dir = Path(fig_path), Path(tiles_dir)
    with __import__("PIL.Image", fromlist=["Image"]).open(fig_path) as opened:
        opened.load()
        source = _rgb_copy(opened)
    rows = []
    for i, top in enumerate(range(0, source.height, row_height)):
        tile = source.crop((0, top, source.width, min(top + row_height, source.height)))
        info = _save_bounded_png(tile, tiles_dir / f"{fig_path.stem}_row_{i + 1:02d}.png",
                                 max_bytes // 2, 1200, row_height)
        rows.append(info)
    return rows


# ---------------------------------------------------------------- ROI 选取
def select_rois(target2d: np.ndarray, size: int = ROI_SIZE, limit: int = ROI_LIMIT):
    """与参考代码一致：前景 mask 上按连通域面积取大 ROI，中心不重叠。"""
    from skimage.measure import regionprops

    binary = target2d > np.percentile(target2d, 75)
    binary[: size // 2, :] = binary[-size // 2:, :] = False
    binary[:, : size // 2] = binary[:, -size // 2:] = False
    props = sorted(regionprops(binary.astype(np.uint8)), key=lambda p: p.area, reverse=True)
    rois, taken = [], []
    height, width = target2d.shape
    for prop in props:
        cy, cx = map(int, prop.centroid)
        if any(abs(cy - y) < size and abs(cx - x) < size for y, x in taken):
            continue
        y0 = min(max(cy - size // 2, 0), height - size)
        x0 = min(max(cx - size // 2, 0), width - size)
        rois.append((y0, x0, size, size))
        taken.append((cy, cx))
        if len(rois) >= limit:
            break
    return rois


# ---------------------------------------------------------------- 图 1-3：定性
def render_pseudocolor_overview(mips: dict[str, np.ndarray], rois, out: Path) -> Path:
    methods = _ordered_methods(mips)
    combined = np.concatenate([mips[m].ravel()[::32] for m in methods])
    lo, hi = np.percentile(combined, OVERVIEW_PCTL)
    fig, axes = plt.subplots(1, len(methods), figsize=(3.0 * len(methods), 3.2), dpi=180)
    axes = np.atleast_1d(axes)
    for col, method in enumerate(methods):
        axes[col].imshow(_normalize(mips[method], lo, hi), cmap=CMAP)
        axes[col].set_title(_label(method), fontsize=8)
        axes[col].axis("off")
        if col == 0:
            for y, x, h, w in rois:
                axes[col].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="white", linewidth=0.8))
    fig.suptitle("Pseudocolor MIP (shared window)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_roi_zooms(mips: dict[str, np.ndarray], target2d: np.ndarray, rois, out: Path) -> Path:
    methods = _ordered_methods(mips)
    fg = target2d > np.percentile(target2d, 75)
    fig, axes = plt.subplots(len(rois), len(methods), figsize=(2.1 * len(methods), 2.2 * len(rois)), dpi=200, squeeze=False)
    for row, (y, x, h, w) in enumerate(rois):
        for col, method in enumerate(methods):
            crop = mips[method][y : y + h, x : x + w]
            lo, hi = np.percentile(crop, ZOOM_PCTL)
            axes[row][col].imshow(_normalize(crop, lo, hi), cmap=CMAP)
            if method != "target":
                from skimage import segmentation

                bd = segmentation.find_boundaries(fg[y : y + h, x : x + w], mode="outer")
                axes[row][col].contour(bd, levels=[0.5], colors="red", linewidths=0.55)
            if row == 0:
                axes[row][col].set_title(_label(method), fontsize=7)
            if col == 0:
                axes[row][col].set_ylabel(f"ROI {row + 1}", fontsize=7)
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])
    fig.suptitle("ROI zooms: pseudocolor MIP, red = target foreground contour", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_patch_rows(patches: dict[str, list[np.ndarray]], out: Path) -> Path:
    """patches[name] = list of 3D (z,y,x) patch；渲染 mid-frame，每列一方法。"""
    methods = _ordered_methods(patches)
    n_patch = len(patches[methods[0]])
    fig, axes = plt.subplots(n_patch, len(methods), figsize=(2.6 * len(methods), 2.7 * n_patch), dpi=180, squeeze=False)
    for row in range(n_patch):
        for col, method in enumerate(methods):
            img = patches[method][row][patches[method][row].shape[0] // 2]
            lo, hi = np.percentile(img, ZOOM_PCTL)
            axes[row][col].imshow(_normalize(img, lo, hi), cmap=CMAP)
            if row == 0:
                axes[row][col].set_title(_label(method), fontsize=8)
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])
    fig.suptitle("Scored ruler patches (top fg-rich, mid z-slice)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- 图 4-5：定量
# C3 常规化：PSNR 之外默认统计 SSIM/SNR/SBR/std_ratio（std_ratio 由 pred_std/target_std 现算）
METRIC_SPECS: list[tuple[str, str, str, int]] = [
    # (列标签, region, json key 或 "std_ratio", 小数位)
    ("PSNR-all", "all", "psnr_db", 2),
    ("PSNR-fg", "foreground", "psnr_db", 2),
    ("PSNR-bg", "background", "psnr_db", 2),
    ("SSIM", "foreground", "ssim", 3),
    ("SNR-fg", "foreground", "snr_db", 2),
    ("SBR-fg", "foreground", "sbr_db", 2),
    ("stdR-fg", "foreground", "std_ratio", 2),
]
_BAR_SPECS = ["PSNR-all", "PSNR-fg", "SSIM", "SNR-fg", "SBR-fg", "stdR-fg"]


def collect_arm_metrics(eval_json: Path) -> dict[str, list[float]]:
    """单个 eval JSON -> {列标签: [值]}；缺失/NaN 一律记 nan（聚合时跳过）。"""
    data = json.loads(Path(eval_json).read_text(encoding="utf-8"))
    regions = data.get("regions", {})
    out: dict[str, list[float]] = {label: [] for label, *_ in METRIC_SPECS}
    for label, region, key, _dec in METRIC_SPECS:
        r = regions.get(region, {})
        if key == "std_ratio":
            ps, ts = r.get("pred_std"), r.get("target_std")
            try:
                val = float(ps) / float(ts) if ps is not None and ts else float("nan")
            except (TypeError, ZeroDivisionError):
                val = float("nan")
        else:
            val = r.get(key, float("nan"))
        try:
            out[label].append(float(val))
        except (TypeError, ValueError):
            out[label].append(float("nan"))
    return out


def _metric_stats(vals: list[float]) -> tuple[float, float] | None:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean()), float((arr.max() - arr.min()) / 2)


def render_metric_bars(matrix: dict[str, dict[str, list[float]]], out: Path) -> Path:
    """matrix[arm][列标签] = list of per-seed 值（collect_matrix_metrics 结构）。"""
    arms = _ordered_methods(matrix)
    specs = [s for s in METRIC_SPECS if s[0] in _BAR_SPECS]
    fig, axes = plt.subplots(1, len(specs), figsize=(2.6 * len(specs), 3.0))
    for ax, (label, _region, _key, dec) in zip(axes, specs):
        means, errs, colors, labels = [], [], [], []
        for arm in arms:
            stats = _metric_stats(matrix[arm].get(label, []))
            if stats is None:
                means.append(0.0); errs.append(0.0)
            else:
                means.append(stats[0]); errs.append(stats[1])
            colors.append(_color(arm))
            labels.append(_label(arm))
        x = np.arange(len(arms))
        bars = ax.bar(x, means, yerr=errs, capsize=3, color=colors, width=0.62,
                      edgecolor="black", linewidth=0.4)
        for bar, v, e in zip(bars, means, errs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + e,
                    f"{v:.{dec}f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.suptitle("Metrics vs reference (mean ± seed half-range)", y=1.04)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_summary_table(matrix: dict[str, dict[str, list[float]]], out: Path) -> Path:
    """matrix[arm][列标签] = list of per-seed 值。"""
    arms = _ordered_methods(matrix)
    labels = [label for label, *_ in METRIC_SPECS if any(label in matrix.get(a, {}) for a in arms)]
    cell_text = []
    for arm in arms:
        row = [_label(arm)]
        for label in labels:
            stats = _metric_stats(matrix[arm].get(label, []))
            dec = next(d for l, _r, _k, d in METRIC_SPECS if l == label)
            row.append("—" if stats is None else
                       (f"{stats[0]:.{dec}f}" if len(matrix[arm][label]) == 1
                        else f"{stats[0]:.{dec}f}±{stats[1]:.{dec}f}"))
        cell_text.append(row)
    fig, ax = plt.subplots(figsize=(1.35 * len(labels) + 1.6, 0.42 * len(arms) + 1.0))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=["Method"] + labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor("#eaeaf2")
            cell.set_text_props(fontweight="bold")
        elif c == 0:
            cell.set_text_props(fontweight="bold", ha="left")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    # dpi=200 常超 1200px（CLAUDE.md 读图限制），表格类直接落一份 ≤1200px 的审查副本
    from PIL import Image
    with Image.open(out) as im:
        if im.width > 1200:
            ratio = 1200.0 / im.width
            im.resize((1200, int(im.height * ratio)), Image.LANCZOS).save(out)
    return out


# ---------------------------------------------------------------- 推理（与评估一致）
def _load_stage2_model(mode_dir: Path, device: torch.nn.Module) -> Any:
    cfg = json.loads((mode_dir / "config.json").read_text(encoding="utf-8"))
    train_cfg = cfg.get("training", cfg)
    sys.path.insert(0, str(VALID_ROOT))
    from models.network import Network_CNR

    model = Network_CNR(in_channels=1, out_channels=1, f_maps=int(train_cfg["base_features"]),
                        n_groups=int(train_cfg["n_groups"])).to(device)
    ckpt = mode_dir / "checkpoints" / f"{mode_dir.name}_best.pth"
    if not ckpt.exists():
        ckpt = mode_dir / "checkpoints" / f"{mode_dir.name}_final.pth"
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def _tiled_predict(model, stack: np.ndarray, mean: float, std: float, device, z_limit: int | None = None) -> np.ndarray:
    """64×128×128 平铺推理（与全栈评估同规格），返回去标准化预测。"""
    z_limit = min(z_limit or stack.shape[0], stack.shape[0])
    out = np.zeros((z_limit, stack.shape[1], stack.shape[2]), dtype=np.float32)
    with torch.no_grad():
        for z0 in range(0, z_limit - 63, 64):
            for y0 in range(0, stack.shape[1] - 127, 128):
                for x0 in range(0, stack.shape[2] - 127, 128):
                    tile = stack[z0 : z0 + 64, y0 : y0 + 128, x0 : x0 + 128]
                    inp = torch.from_numpy((tile - mean) / std).float().unsqueeze(0).unsqueeze(0).to(device)
                    out[z0 : z0 + 64, y0 : y0 + 128, x0 : x0 + 128] = (
                        model(inp).squeeze().cpu().numpy().astype(np.float32) * std + mean
                    )
    return out


def collect_stage2_outputs(run_dir: Path | None, z_limit: int | None = None,
                           arm_map: dict[str, Path] | None = None) -> dict[str, Any]:
    """单 run：加载 nf 栈 → 各臂 best 平铺推理 → MIP + ruler patch 中面。

    arm_map 非空时为跨 run 单臂对比模式：label -> mode_dir（含 config.json/checkpoints），
    配置取第一个 arm 的 config.json（须与其它 arm 同数据集/同 patch 规格）。
    """
    if arm_map:
        cfg = json.loads((next(iter(arm_map.values())) / "config.json").read_text(encoding="utf-8"))
    else:
        cfg = json.loads((run_dir / "valid" / "config.json").read_text(encoding="utf-8"))
    train_cfg = cfg["training"]
    stack = tifffile.imread(sorted(Path(train_cfg["train_folder"]).glob("*.tif"))[0]).astype(np.float32)
    mean, std = float(stack.mean()), float(stack.std())
    target_path = cfg["evaluation"].get("mean9_proxy")
    target = tifffile.imread(target_path).astype(np.float32) if target_path else None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if arm_map:
        arms = list(arm_map.keys())
    else:
        arms = [m for m in ("valid", "sn2n_xy", "iso3d", "qmask") if (run_dir / m).exists()]
    mips: dict[str, np.ndarray] = {"noisy": stack[: z_limit or stack.shape[0]].max(axis=0)}
    patch_maps: dict[str, list[np.ndarray]] = {}

    # ruler patch 坐标：复用 eval_metrics.json 里记录的 selected_patch_indices
    # （arm_map 模式取第一个 arm 自己的 eval_metrics.json）
    eval_json = (arm_map[next(iter(arm_map))] / "metrics" / "eval_metrics.json") if arm_map \
        else run_dir / "valid" / "metrics" / "eval_metrics.json"
    positions = json.loads(eval_json.read_text(encoding="utf-8"))["selected_patch_indices"][:3]
    from datasets.dataset_fs import ReadDatasets

    # 与 qmask/train.py 完全一致地建数据集(patch_num=-1 全网格);
    # selected_patch_indices 是在该全网格 enumerate 下的位置,直接用作 dataset 下标。
    dataset = ReadDatasets(
        dataPath=train_cfg["train_folder"], mode="train", dataType="3D", dataExtension="tif",
        z_patch=int(train_cfg["z_patch"]), w_patch=int(train_cfg["w_patch"]), h_patch=int(train_cfg["h_patch"]),
        z_overlap=float(train_cfg["z_overlap"]), w_overlap=float(train_cfg["w_overlap"]),
        h_overlap=float(train_cfg["h_overlap"]), patch_num=int(train_cfg.get("patch_num", -1)),
        dataNum=int(train_cfg.get("train_frame_num", 10000)),
    )
    patch_maps["noisy"] = [dataset[i][0].squeeze(0).numpy().astype(np.float32) for i in positions]
    if target is not None:
        target3d = target if target.ndim == 3 else None
        if target3d is None:
            patch_maps["target"] = [
                np.broadcast_to(target[dataset[i][3] : dataset[i][3] + int(train_cfg["w_patch"]),
                                       dataset[i][4] : dataset[i][4] + int(train_cfg["h_patch"])],
                                (int(train_cfg["z_patch"]), int(train_cfg["w_patch"]), int(train_cfg["h_patch"]))).astype(np.float32)
                for i in positions
            ]
        else:
            patch_maps["target"] = [target3d[dataset[i][2] : dataset[i][2] + int(train_cfg["z_patch"]),
                                             dataset[i][3] : dataset[i][3] + int(train_cfg["w_patch"]),
                                             dataset[i][4] : dataset[i][4] + int(train_cfg["h_patch"])] for i in positions]

    for arm in arms:
        model = _load_stage2_model(arm_map[arm] if arm_map else run_dir / arm, device)
        pred = _tiled_predict(model, stack, mean, std, device, z_limit=z_limit)
        mips[arm] = pred.max(axis=0)
        patch_maps[arm] = []
        for i in positions:
            _, z0, y0, x0 = (int(v) for v in dataset[i][1:])
            tile = stack[z0 : z0 + int(train_cfg["z_patch"]), y0 : y0 + int(train_cfg["w_patch"]), x0 : x0 + int(train_cfg["h_patch"])]
            inp = torch.from_numpy((tile - mean) / std).float().unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(inp).squeeze().cpu().numpy().astype(np.float32) * std + mean
            patch_maps[arm].append(out)
        del model
        torch.cuda.empty_cache()
        print(f"  [paperviz] {arm} inferred")
    return {"mips": mips, "patch_maps": patch_maps, "target2d": target if (target is not None and target.ndim == 2) else mips["target"] if "target" in mips else None}


# ---------------------------------------------------------------- 矩阵汇总
def collect_matrix_metrics(matrix_root: Path, seeds: list[str],
                           arms: list[str]) -> dict[str, dict[str, list[float]]]:
    """matrix[arm][列标签] = list of per-seed 值（C3：全指标）。"""
    matrix: dict[str, dict[str, list[float]]] = {}
    for arm in arms:
        matrix[arm] = {label: [] for label, *_ in METRIC_SPECS}
        for seed in seeds:
            path = matrix_root / f"stage02_nf_matrix4_seed{seed}" / arm / "metrics" / "eval_metrics.json"
            for label, vals in collect_arm_metrics(path).items():
                matrix[arm][label].extend(vals)
    return matrix


# ---------------------------------------------------------------- 驱动
def _emit(fig_paths: list[Path], out_dir: Path) -> None:
    tiles_root = out_dir / "review"
    html = ["<html><body style='font-family:sans-serif'><h3>paperviz review (只读这些切块)</h3>"]
    for fig in fig_paths:
        rows = make_figure_tiles(fig, tiles_root)
        html.append(f"<h4>{fig.name}</h4>")
        for info in rows:
            html.append(f"<img src='review/{Path(info['path']).name}' style='max-width:1200px;display:block;margin:4px 0'>")
    (out_dir / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(f"[paperviz] review tiles + index.html -> {out_dir / 'index.html'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="固化 paper-style 可视化（定性+定量+review 切块）")
    parser.add_argument("--run-dir", help="stage2 run 目录（多臂定性）")
    parser.add_argument("--matrix-root", help="含 stage02_nf_matrix4_seed* 的 runs 根目录")
    parser.add_argument("--seeds", default="3407,3408,3409")
    parser.add_argument("--z-limit", type=int, default=512, help="MIP 推理的 z 帧数上限")
    parser.add_argument("--out", default=None)
    parser.add_argument("--arm", action="append", default=None,
                        help="跨 run 单臂对比定性：label=mode_dir（可多次，如 --arm ctrl=.../qmask --arm wr5e-5=.../qmask）")
    args = parser.parse_args(argv)

    arm_map = None
    if args.arm:
        arm_map = {}
        for item in args.arm:
            label, _, path = item.partition("=")
            if not label or not path:
                parser.error(f"--arm 格式应为 label=mode_dir，收到 {item!r}")
            arm_map[label.strip()] = Path(path.strip())

    if arm_map:
        out_dir = Path(args.out) if args.out else next(iter(arm_map.values())).parent / "paperviz_arms"
        matrix_root = None
    elif args.matrix_root:
        matrix_root = Path(args.matrix_root)
        out_dir = Path(args.out) if args.out else matrix_root / "paperviz"
    elif args.run_dir:
        matrix_root = None
        out_dir = Path(args.out) if args.out else Path(args.run_dir) / "paperviz"
    else:
        parser.error("需要 --run-dir 或 --matrix-root")
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [s.strip() for s in args.seeds.split(",")]

    figs: list[Path] = []

    # 定量（矩阵）
    matrix_metrics = None
    if matrix_root:
        arms = ["valid", "sn2n_xy", "iso3d", "qmask"]
        matrix_metrics = collect_matrix_metrics(matrix_root, seeds, arms)
        figs.append(render_metric_bars(matrix_metrics, out_dir / "metric_bars.png"))
        figs.append(render_summary_table(matrix_metrics, out_dir / "summary_table.png"))
        run_dir = matrix_root / f"stage02_nf_matrix4_seed{seeds[0]}"
    elif arm_map:
        # C3：跨 run 单臂对比也出定量表（aligned final 优先，缺则用 final）
        arm_metrics: dict[str, dict[str, list[float]]] = {}
        for label, mdir in arm_map.items():
            for name in ("eval_metrics_aligned.json", "eval_metrics.json"):
                p = mdir / "metrics" / name
                if p.exists():
                    arm_metrics[label] = collect_arm_metrics(p)
                    print(f"[paperviz] {label}: {p.name}")
                    break
            else:
                print(f"[paperviz] 警告：{label} 无 eval_metrics(.aligned).json，定量表跳过该臂")
        if arm_metrics:
            figs.append(render_summary_table(arm_metrics, out_dir / "metrics_arms.png"))
    else:
        run_dir = Path(args.run_dir) if args.run_dir else None

    # 定性（第一个 seed 的 run，或 --arm 跨 run 对比）
    if arm_map:
        print(f"[paperviz] arm-map qualitative: {', '.join(arm_map)} (z_limit={args.z_limit})")
        outputs = collect_stage2_outputs(None, z_limit=args.z_limit, arm_map=arm_map)
    else:
        print(f"[paperviz] inferring qualitative arms from {run_dir} (z_limit={args.z_limit})")
        outputs = collect_stage2_outputs(run_dir, z_limit=args.z_limit)
    mips = {k: v for k, v in outputs["mips"].items()}
    target2d = outputs["target2d"]
    if target2d is not None:
        mips["target"] = target2d
    rois = select_rois(target2d)
    figs.append(render_pseudocolor_overview(mips, rois, out_dir / "pseudocolor_overview.png"))
    figs.append(render_roi_zooms(mips, target2d, rois, out_dir / "roi_zooms.png"))
    figs.append(render_patch_rows(outputs["patch_maps"], out_dir / "patch_rows.png"))
    (out_dir / "selected_rois.json").write_text(
        json.dumps([{"y": y, "x": x, "h": h, "w": w} for y, x, h, w in rois], indent=2), encoding="utf-8")

    _emit(figs, out_dir)
    for f in figs:
        print(f"[paperviz] {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
