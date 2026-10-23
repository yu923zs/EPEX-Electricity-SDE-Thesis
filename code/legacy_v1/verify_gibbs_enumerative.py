# -*- coding: utf-8 -*-
"""
Gibbs 采样器逐变量补抽的可枚举后验验证

在固定 σ², τ², λ (取先验均值)、小规模可枚举模型下，
对每个 θ 配置计算边际似然（解析积分 β_j），得到精确 PIP，
与 Gibbs 采样器的 PIP 对比。

验证：θ_j 的边际更新是否使用正确的贝叶斯因子 O_j，
      β_j 是否从条件后验 N(μ_j, V_j) 抽样。
"""
import os
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_stage_B import gibbs_ss

# ---- 固定参数 ----
K = 50          # 代表态数
p = 3           # 候选变量数（2^3=8 种 θ 配置可枚举）
pi0 = 0.5
sigma2 = 1.0
tau2 = 4.0
sg2 = 100.0     # 截距先验方差

rng = np.random.default_rng(42)
# 设计矩阵 + 真实系数
Phi = rng.standard_normal((K, p))
beta_true = np.array([1.5, 0.0, -2.0])  # 只有变量 0 和 2 非零
gamma_true = 0.3
y = gamma_true + Phi @ beta_true + 0.5 * rng.standard_normal(K)
w = np.ones(K)

# ---- 可枚举后验：固定 σ², τ², λ=1, w=1 ----
# 对每个 θ ∈ {0,1}^p，积分 β|θ 得到边际似然 p(y|θ)
# p(y|θ) = N(y; γ1, σ² I + Φ_A τ² Φ_A^T) 的高斯积分
# 取 γ 的 N(0, sg2) 先验并积分

def log_marginal_likelihood(theta):
    """对给定 θ，积分 γ 和 β_A 得到 log p(y|θ)。"""
    A = int(theta.sum())
    ZA = np.column_stack([np.ones(K), Phi[:, theta]])
    # 先验协方差：γ ~ N(0,sg2), β_A ~ N(0, τ² I)
    V0 = np.diag([sg2] + [tau2] * A)
    # 数据协方差：σ² I + ZA V0 ZA^T
    Sigma = sigma2 * np.eye(K) + ZA @ V0 @ ZA.T
    # y ~ N(0, Sigma)
    sign, logdet = np.linalg.slogdet(Sigma)
    return -0.5 * (K * np.log(2 * np.pi) + logdet
                   + y @ np.linalg.solve(Sigma, y))

# 枚举所有 2^p 配置
configs = []
log_ml = []
for i in range(2 ** p):
    theta = np.array([(i >> j) & 1 for j in range(p)], dtype=bool)
    configs.append(theta)
    log_ml.append(log_marginal_likelihood(theta))

log_ml = np.array(log_ml)
# 归一化后验：p(θ|y) ∝ p(y|θ) * π0^{|θ|} (1-π0)^{p-|θ|}
log_prior = np.array([np.sum(np.log(pi0) * c + np.log(1 - pi0) * (1 - c))
                      for c in configs])
log_post = log_ml + log_prior
log_post -= np.max(log_post)
post = np.exp(log_post)
post /= post.sum()

# 精确 PIP
pip_exact = np.zeros(p)
for c, pr in zip(configs, post):
    pip_exact += pr * c.astype(float)

# ---- Gibbs PIP ----
prior = {"a_lam": 2.0, "b_lam": 2.0, "a_sig": 1.0, "b_sig": 1.0,
         "a_tau": 1.0, "b_tau": 1.0, "pi0": pi0, "sigma_gamma2": sg2}
# 用 normal likelihood 使 λ=1 固定，匹配可枚举设定
trace, final = gibbs_ss(y, Phi, w, prior, n_burn=5000, n_post=20000,
                        seed=123, likelihood="normal")
theta_samp = trace["theta"]
pip_gibbs = theta_samp.mean(axis=0)

print("=" * 60)
print("Gibbs 逐变量补抽可枚举后验验证")
print("=" * 60)
print(f"K={K}, p={p}, 2^p={2**p} 配置枚举")
print(f"真实 β = {beta_true}, γ = {gamma_true}")
print()
print(f"{'变量':>6} {'精确 PIP':>12} {'Gibbs PIP':>12} {'绝对差':>10}")
print("-" * 45)
for j in range(p):
    diff = abs(pip_exact[j] - pip_gibbs[j])
    print(f"{j:>6} {pip_exact[j]:>12.4f} {pip_gibbs[j]:>12.4f} {diff:>10.4f}")

max_diff = np.max(np.abs(pip_exact - pip_gibbs))
print()
print(f"最大 PIP 偏差: {max_diff:.4f}")
print(f"阈值 (0.05): {'通过' if max_diff < 0.05 else '未通过'}")

# ---- 系数矩验证 ----
# 对每个 θ 配置，β 的条件后验均值和方差
beta_post_mean_exact = np.zeros(p)
for c, pr in zip(configs, post):
    if c.sum() == 0:
        continue
    ZA = np.column_stack([np.ones(K), Phi[:, c]])
    V0 = np.diag([sg2] + [tau2] * int(c.sum()))
    Sigma = sigma2 * np.eye(K) + ZA @ V0 @ ZA.T
    # β_A 的后验均值：V0 ZA^T (σ²I + ZA V0 ZA^T)^{-1} y
    V0ZA = ZA @ V0  # (K, A+1)
    beta_A_mean = V0.T @ ZA.T @ np.linalg.solve(Sigma, y)  # (A+1,)
    idx = np.where(c)[0]
    for k, j in enumerate(idx):
        beta_post_mean_exact[j] += pr * beta_A_mean[k + 1]

beta_gibbs_mean = trace["beta"].mean(axis=0)
print()
print(f"{'变量':>6} {'精确 E[β]':>12} {'Gibbs E[β]':>12} {'绝对差':>10}")
print("-" * 45)
for j in range(p):
    diff = abs(beta_post_mean_exact[j] - beta_gibbs_mean[j])
    print(f"{j:>6} {beta_post_mean_exact[j]:>12.4f} {beta_gibbs_mean[j]:>12.4f} {diff:>10.4f}")

max_beta_diff = np.max(np.abs(beta_post_mean_exact - beta_gibbs_mean))
print()
print(f"最大 E[β] 偏差: {max_beta_diff:.4f}")
print(f"阈值 (0.1): {'通过' if max_beta_diff < 0.1 else '未通过'}")
print()
print("代码行参考 (run_stage_B.py):")
print("  - _sample_theta_and_beta() L232-278: 逐变量 Gibbs 更新")
print("  - L241: r_adj = r + phij * beta[j]  (移除 β_j 影响)")
print("  - L242-243: a_j = phij^T D^{-1} phij, c_j = phij^T D^{-1} r_adj")
print("  - L244-251: log O_j = log(π/(1-π)) - 0.5*log(1+a_j τ²) + τ² c_j²/(2(1+a_j τ²))")
print("  - L259: p_in = sigmoid(log O_j)")
print("  - L266-268: β_j | θ_j=1 ~ N(τ² c_j/(1+a_j τ²), τ²/(1+a_j τ²))")
print("  - L275/278: 用新 β_j 更新残差 r")
