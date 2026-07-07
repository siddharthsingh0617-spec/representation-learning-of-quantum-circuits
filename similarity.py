

import numpy as np
from scipy.linalg import qr


def _center_gram(K):
  
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_hsic(K, L):

    Kc = _center_gram(K)
    Lc = _center_gram(L)
    n = K.shape[0]
    return np.trace(Kc @ Lc) / ((n - 1) ** 2)


def linear_cka(X, Y):

    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    K = X @ X.T
    L = Y @ Y.T
    hsic_xy = linear_hsic(K, L)
    hsic_xx = linear_hsic(K, K)
    hsic_yy = linear_hsic(L, L)
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def rbf_kernel(X, sigma_frac=0.4):
   
    sq_dists = np.sum(X ** 2, axis=1, keepdims=True) \
        + np.sum(X ** 2, axis=1) - 2 * X @ X.T
    sq_dists = np.maximum(sq_dists, 0)
    median_dist = np.median(np.sqrt(sq_dists[sq_dists > 0])) \
        if np.any(sq_dists > 0) else 1.0
    sigma = sigma_frac * median_dist
    if sigma < 1e-12:
        sigma = 1.0
    return np.exp(-sq_dists / (2 * sigma ** 2))


def rbf_cka(X, Y, sigma_frac=0.4):
 
    K = rbf_kernel(X, sigma_frac)
    L = rbf_kernel(Y, sigma_frac)
    hsic_xy = linear_hsic(K, L)
    hsic_xx = linear_hsic(K, K)
    hsic_yy = linear_hsic(L, L)
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def linear_regression_r2(X, Y):
  
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    Qy, _ = qr(Y, mode="economic")
    num = np.linalg.norm(Qy.T @ X, "fro") ** 2
    den = np.linalg.norm(X, "fro") ** 2
    if den < 1e-12:
        return 0.0
    return float(num / den)


def cca_mean_correlation(X, Y):

    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    Qx, Rx = qr(X, mode="economic")
    Qy, Ry = qr(Y, mode="economic")
    # guard against rank deficiency
    if np.linalg.matrix_rank(Rx) < Rx.shape[1] or \
       np.linalg.matrix_rank(Ry) < Ry.shape[1]:
        Qx = Qx[:, :np.linalg.matrix_rank(Rx)]
        Qy = Qy[:, :np.linalg.matrix_rank(Ry)]
    M = Qx.T @ Qy
    svals = np.linalg.svd(M, compute_uv=False)
    svals = np.clip(svals, -1, 1)
    return float(np.mean(svals))


def all_similarity_metrics(X, Y, rbf_sigma_frac=0.4):
   
    return {
        "linear_cka": linear_cka(X, Y),
        "rbf_cka": rbf_cka(X, Y, rbf_sigma_frac),
        "linear_reg_r2": linear_regression_r2(X, Y),
        "cca_mean_corr": cca_mean_correlation(X, Y),
    }


if __name__ == "__main__":

    rng = np.random.default_rng(0)
    n, d = 50, 8

    X = rng.normal(size=(n, d))

    print("=== Sanity checks ===")
    # 1. CKA(X, X) == 1
    print(f"linear_cka(X, X)  = {linear_cka(X, X):.6f}  (expect ~1.0)")
    print(f"rbf_cka(X, X)     = {rbf_cka(X, X):.6f}  (expect ~1.0)")

   
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    print(f"linear_cka(X, XQ) = {linear_cka(X, X @ Q):.6f}  (expect ~1.0, orthogonal invariance)")

    print(f"linear_cka(X, 5X) = {linear_cka(X, 5 * X):.6f}  (expect ~1.0, isotropic scale invariance)")

    D = np.diag(rng.uniform(0.1, 5, size=d))
    print(f"linear_cka(X, XD) = {linear_cka(X, X @ D):.6f}  (expect <1.0, non-isotropic scaling)")

    Y_indep = rng.normal(size=(n, d))
    print(f"linear_cka(X, Y_indep) = {linear_cka(X, Y_indep):.6f}  (expect low, near 0)")

    print(f"\nrho_bar_cca(X, X) = {cca_mean_correlation(X, X):.6f}  (expect ~1.0)")
    print(f"R2(X, X) = {linear_regression_r2(X, X):.6f}  (expect ~1.0)")
