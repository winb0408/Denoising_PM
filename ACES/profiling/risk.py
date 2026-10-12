"""Pair-Risk Field 的三个核心风险估计器（修正方案 §8-§10）。

所有估计器输出都在 [0,1] 风险空间（越高越差）：

1. Noise Risk R_N（§8.1，repeat 分支）::

       Ȳ = mean_k(Y_k),  N̂_k = Y_k − Ȳ
       ρ_N(Δ) = corr(N̂_k(r), N̂_k(r+Δ))   # 对 (k, r) 池化
       R_N(Δ) = |ρ_N(Δ)|
       R_NV(Δ) = |log(σ²_N(r+Δ) / σ²_N(r))|  # 方差一致性，clip 到 [0,1]
       R_N*(Δ) = α_N·R̃_N + (1−α_N)·R̃_NV

2. Structure Risk R_S（§9）::

       S_LF = G_σ(Y)
       R_S1 = 1 − corr(S_LF(r), S_LF(r+Δ))
       R_S2 = 1 − SSIM_local   # 第一版用 0.5 常数 stub（见 TODO）
       R_S3 = 1 − corr(∇S(r), ∇S(r+Δ))   # np.gradient，逐分量相关取平均
       R_S = 0.7·R̃_S1 + 0.3·R̃_S3
       # 注：省略 local SSIM 后，权重由 0.5/0.3/0.2 重新归一化为 0.7/0.3。

3. Identity / Shortcut Risk R_I（§10）::

       nMSE(Δ) = E[(Y(r)−Y(r+Δ))²] / (Var(Y)+ε)
       R_I(Δ) = exp(−nMSE/τ_I) · max(ρ_Y(Δ), 0)

所有 pair 统计在“有效内部”体素上计算：边界处直接裁剪（valid-crop），
不做任何 padding。体积较大时用 ``stride`` 做确定性步长子采样以控制内存。
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

OffsetZyx = tuple[int, int, int]


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def shifted_pair(
    volume: np.ndarray,
    offset_zyx: OffsetZyx,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray] | None:
    """返回 (volume[r], volume[r+Δ]) 的有效内部体素对。

    边界处直接裁剪（valid-crop），不做 padding；``stride`` 为确定性
    步长子采样（沿每个空间轴隔 stride 取一个体素）以控制内存。
    若裁剪后有效体素不足 2 个，返回 None。
    """
    if stride < 1:
        raise ValueError(f"stride 必须 >= 1，收到 {stride}")
    src_slices: list[slice] = []
    dst_slices: list[slice] = []
    for size, delta in zip(volume.shape, offset_zyx):
        lo = max(0, -int(delta))
        hi = min(int(size), int(size) - int(delta))
        if hi - lo < 2:
            return None
        src_slices.append(slice(lo, hi, stride))
        dst_slices.append(slice(lo + int(delta), hi + int(delta), stride))
    return volume[tuple(src_slices)], volume[tuple(dst_slices)]


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    """池化 Pearson 相关（零均值归一后内积），方差为 0 时返回 0。"""
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    if a.size < 2 or a.size != b.size:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(float(np.mean(a * a)) * float(np.mean(b * b)))
    if denom <= 1e-12:
        return 0.0
    return float(np.mean(a * b) / denom)


def _clip01(value: float) -> float:
    """裁剪到 [0,1]；非有限值保持 NaN 以便上游报告。"""
    if not np.isfinite(value):
        return float("nan")
    return float(np.clip(value, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 1. Noise Risk（repeat 分支）
# ---------------------------------------------------------------------------

class NoiseRiskEstimator:
    """基于 K 次重复采集的噪声风险 R_N（§8.1.1 / §8.1.2）。

    输入 repeat stack [K,Z,Y,X]（float32）。先算 Ȳ 与残差 N̂_k = Y_k − Ȳ，
    再对每个 offset 计算 ρ_N / R_N / R_NV / R_N*。

    注意：K 较小（如 9）时逐体素 σ²_N(r) 估计噪声极大（8 自由度的
    卡方差，|log ratio| 均值 ~1.8，会把 R_NV 饱和到 1）。因此默认对
    方差图做高斯平滑（``var_sigma``，局部方差估计）后再计算 R_NV；
    ``var_sigma=0`` 关闭平滑。
    """

    def __init__(self, stride: int = 1, eps: float = 1e-6, var_sigma: float = 2.0) -> None:
        self.stride = int(stride)
        self.eps = float(eps)
        self.var_sigma = float(var_sigma)
        self.mean_: np.ndarray | None = None
        self.residuals_: np.ndarray | None = None
        self.var_map_: np.ndarray | None = None

    def fit(self, stack: np.ndarray) -> "NoiseRiskEstimator":
        data = np.asarray(stack, dtype=np.float32)
        if data.ndim != 4:
            raise ValueError(f"repeat stack 应为 [K,Z,Y,X] 4D，收到 shape {data.shape}")
        if data.shape[0] < 2:
            raise ValueError("Noise Risk 需要至少 2 个 repeat")
        self.mean_ = data.mean(axis=0, dtype=np.float32)
        # N̂_k = Y_k − Ȳ（每个体素上跨 k 已零均值）
        self.residuals_ = data - self.mean_[None, ...]
        # 逐体素噪声方差 σ²_N(r)（无偏估计，K−1 归一），
        # 再做局部平滑以压低小 K 下的估计方差（见类 docstring）
        var_map = self.residuals_.var(axis=0, ddof=1)
        if self.var_sigma > 0:
            var_map = ndimage.gaussian_filter(var_map, sigma=self.var_sigma)
        self.var_map_ = np.asarray(var_map, dtype=np.float32)
        return self

    def _require_fit(self) -> None:
        if self.residuals_ is None or self.var_map_ is None:
            raise RuntimeError("必须先调用 fit(stack) 再计算风险")

    def risk(self, offset_zyx: OffsetZyx, alpha: float = 0.5) -> dict[str, float]:
        """对 offset Δ 计算 R_N 分量字典。

        返回 keys：rn（|ρ_N| clip [0,1]）、rn_corr（带符号 ρ_N）、
        rnv（方差一致性，mean |log ratio| clip [0,1]）、
        rn_star（α·R̃_N + (1−α)·R̃_NV）、rn_samples。
        """
        self._require_fit()
        assert self.residuals_ is not None and self.var_map_ is not None
        # 残差是 [K,Z,Y,X] 4D：K 轴不位移，offset 只作用于后 3 个空间轴
        stack_offset = (0, int(offset_zyx[0]), int(offset_zyx[1]), int(offset_zyx[2]))
        # 残差对 (k, r) 池化做 Pearson
        noise_pair = shifted_pair(self.residuals_, stack_offset, stride=self.stride)
        var_pair = shifted_pair(self.var_map_, offset_zyx, stride=self.stride)
        if noise_pair is None or var_pair is None:
            raise ValueError(f"offset {offset_zyx} 相对体数据过大，无有效重叠")
        left_noise, right_noise = noise_pair
        rho_n = _pearson(left_noise, right_noise)
        rn = _clip01(abs(rho_n))
        # R_NV = |log(σ²(r+Δ)/σ²(r))|，对有效体素取平均后 clip 到 [0,1]
        left_var, right_var = var_pair
        log_ratio = np.log(
            (np.asarray(right_var, dtype=np.float64) + self.eps)
            / (np.asarray(left_var, dtype=np.float64) + self.eps)
        )
        rnv = _clip01(float(np.mean(np.abs(log_ratio))))
        rn_star = _clip01(alpha * rn + (1.0 - alpha) * rnv)
        return {
            "rn": rn,
            "rn_corr": float(rho_n),
            "rnv": rnv,
            "rn_star": rn_star,
            "rn_samples": int(left_noise.size),
        }


# ---------------------------------------------------------------------------
# 2. Structure Risk
# ---------------------------------------------------------------------------

class StructureRiskEstimator:
    """结构风险 R_S（§9）：低频结构相关 + 梯度一致（SSIM 为 stub）。

    第一版省略 local SSIM（stub 返回常数 0.5，见 :meth:`_local_ssim`），
    权重由 0.5/0.3/0.2 重新归一化为 0.7/0.3：
    ``R_S = 0.7·R̃_S1 + 0.3·R̃_S3``。
    """

    # 权重（省略 SSIM 后重新归一化；TODO: 实现 local SSIM 后改回 0.5/0.3/0.2）
    W_S1 = 0.7
    W_S3 = 0.3

    def __init__(self, sigma: float = 0.8, stride: int = 1) -> None:
        self.sigma = float(sigma)
        self.stride = int(stride)
        self.s_lf_: np.ndarray | None = None
        self.grads_: list[np.ndarray] | None = None

    def fit(self, volume: np.ndarray) -> "StructureRiskEstimator":
        data = np.asarray(volume, dtype=np.float32)
        if data.ndim != 3:
            raise ValueError(f"structure volume 应为 [Z,Y,X] 3D，收到 shape {data.shape}")
        # S_LF = G_σ(Y)：低频结构代理
        self.s_lf_ = ndimage.gaussian_filter(data, sigma=self.sigma)
        # ∇S = np.gradient 的三个分量（z/y/x）
        self.grads_ = [np.gradient(self.s_lf_, axis=axis) for axis in range(3)]
        return self

    def _require_fit(self) -> None:
        if self.s_lf_ is None or self.grads_ is None:
            raise RuntimeError("必须先调用 fit(volume) 再计算风险")

    def _local_ssim(self, left: np.ndarray, right: np.ndarray) -> float:
        """Local SSIM stub：暂未实现，返回常数 0.5（中性值）。

        TODO(M1 后续): 实现滑窗局部 SSIM（如 7×7 高斯窗），届时把权重
        恢复为 0.5/0.3/0.2。当前 R_S 已按 0.7/0.3 重新归一化。
        """
        return 0.5

    def risk(self, offset_zyx: OffsetZyx) -> dict[str, float]:
        """对 offset Δ 计算 R_S 分量字典。

        返回 keys：rs、rs_s1（1−ρ_LF）、rs_ssim（stub 常数 0.5）、
        rs_s3（1−mean 逐分量梯度相关）、rs_lf_corr（带符号 ρ_LF）。
        """
        self._require_fit()
        assert self.s_lf_ is not None and self.grads_ is not None
        lf_pair = shifted_pair(self.s_lf_, offset_zyx, stride=self.stride)
        if lf_pair is None:
            raise ValueError(f"offset {offset_zyx} 相对体数据过大，无有效重叠")
        left_lf, right_lf = lf_pair
        rho_lf = _pearson(left_lf, right_lf)
        rs1 = _clip01(1.0 - rho_lf)
        # R_S3：∇S 逐分量（z/y/x）相关后取平均
        component_corrs: list[float] = []
        for grad in self.grads_:
            grad_pair = shifted_pair(grad, offset_zyx, stride=self.stride)
            if grad_pair is None:
                raise ValueError(f"offset {offset_zyx} 相对体数据过大，无有效重叠")
            component_corrs.append(_pearson(grad_pair[0], grad_pair[1]))
        rs3 = _clip01(1.0 - float(np.mean(component_corrs)))
        rs_ssim = self._local_ssim(left_lf, right_lf)
        # 省略 SSIM 后重新归一化的加权（见类 docstring）
        rs = _clip01(self.W_S1 * rs1 + self.W_S3 * rs3)
        return {
            "rs": rs,
            "rs_s1": rs1,
            "rs_ssim": rs_ssim,
            "rs_s3": rs3,
            "rs_lf_corr": float(rho_lf),
        }


# ---------------------------------------------------------------------------
# 3. Identity / Shortcut Risk
# ---------------------------------------------------------------------------

class ShortcutRiskEstimator:
    """Identity / shortcut 风险 R_I（§10）。

    Y(r) ≈ Y(r+Δ) 时模型易学到 Ŝ ≈ Y（identity shortcut）。
    ``R_I = exp(−nMSE/τ_I) · max(ρ_Y, 0)``，乘 ρ_Y 项用于降低
    平坦背景的误判（纯平坦区 nMSE≈0 但 ρ_Y 也无定义/低相关）。
    """

    def __init__(self, tau_i: float | None = None, stride: int = 1, eps: float = 1e-6) -> None:
        # tau_i=None → 数据自适应标定：τ_I = Var(Y) 尺度无关化后的 nMSE 参考值。
        # 固定 0.05 在噪声主导数据上会饱和（nMSE≈1.2–1.5 → exp(−nMSE/0.05)≈0，
        # M1 冒烟实测）；τ_I=1.0 则 nMSE 直接进入指数（≈原始定义的软化版）。
        # Phase 1 sensitivity analysis 会统一复核该参数（PLAN.md §3）。
        self.tau_i = float(tau_i) if tau_i is not None else 1.0
        self.stride = int(stride)
        self.eps = float(eps)
        self.volume_: np.ndarray | None = None
        self.var_y_: float | None = None

    def fit(self, volume: np.ndarray) -> "ShortcutRiskEstimator":
        data = np.asarray(volume, dtype=np.float32)
        if data.ndim != 3:
            raise ValueError(f"volume 应为 [Z,Y,X] 3D，收到 shape {data.shape}")
        self.volume_ = data
        self.var_y_ = float(np.asarray(data, dtype=np.float64).var())
        return self

    def _require_fit(self) -> None:
        if self.volume_ is None or self.var_y_ is None:
            raise RuntimeError("必须先调用 fit(volume) 再计算风险")

    def risk(self, offset_zyx: OffsetZyx) -> dict[str, float]:
        """对 offset Δ 计算 R_I 分量字典。

        返回 keys：ri、ri_nmse、ri_corr_y（带符号 ρ_Y）。
        """
        self._require_fit()
        assert self.volume_ is not None and self.var_y_ is not None
        pair = shifted_pair(self.volume_, offset_zyx, stride=self.stride)
        if pair is None:
            raise ValueError(f"offset {offset_zyx} 相对体数据过大，无有效重叠")
        left, right = pair
        left64 = np.asarray(left, dtype=np.float64)
        right64 = np.asarray(right, dtype=np.float64)
        # nMSE(Δ) = E[(Y(r)−Y(r+Δ))²] / (Var(Y)+ε)
        nmse = float(np.mean((left64 - right64) ** 2) / (self.var_y_ + self.eps))
        rho_y = _pearson(left64, right64)
        ri = float(np.exp(-nmse / self.tau_i) * max(rho_y, 0.0))
        return {
            "ri": _clip01(ri),
            "ri_nmse": nmse,
            "ri_corr_y": float(rho_y),
        }
