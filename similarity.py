"""
Representation similarity metrics, implementing the methods from
Kornblith, Norouzi, Lee & Hinton (2019), "Similarity of Neural Network
Representations Revisited" (ICML 2019), adapted here to compare
*quantum* circuit representations instead of classical neural-network
activations.

Given two representation matrices X (n_samples x d1) and Y (n_samples x d2)
-- here, layer-wise Pauli expectation-value vectors extracted from a
quantum circuit for a batch of n_samples inputs -- we compute:

  - Linear CKA   (Eq. 4 with linear kernel, Section 3 of the paper)
  - RBF CKA      (Eq. 4 with RBF kernel)
  - Linear regression R^2 (Eq. 16)
  - CCA mean correlation rho-bar (Eq. 8)

All formulas are implemented directly from the paper's equations rather
than via a third-party CKA package, so the implementation is fully
auditable.
"""

import numpy as np
from scipy.linalg import qr


def _center_gram(K):
    """H K H, the double-centering operation (Eq. 3), H = I - (1/n) 11^T."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_hsic(K, L):
    """Unbiased-flavor empirical HSIC, Eq. 3 (we use the simple centered
    estimator, consistent with the paper's main text)."""
    Kc = _center_gram(K)
    Lc = _center_gram(L)
    n = K.shape[0]
    return np.trace(Kc @ Lc) / ((n - 1) ** 2)


def linear_cka(X, Y):
    """Linear CKA (Eq. 4 with linear kernel K = X X^T, L = Y Y^T).
    Equivalent closed form: ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F),
    Eq. 14's denominator structure -- we use the direct Gram-matrix
    form here for numerical clarity and to reuse linear_hsic."""
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
    """RBF kernel k(x_i, x_j) = exp(-||x_i-x_j||^2 / (2*sigma^2)) with
    sigma set as a fraction of the median pairwise distance, exactly
    as specified in Section 4 ('Kernel Selection') of the paper."""
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
    """RBF CKA (Eq. 4 with RBF kernel), Kernel Selection paragraph."""
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
    """R^2 of fitting X via linear regression on Y (Eq. 16):
    R^2 = ||Q_Y^T X||_F^2 / ||X||_F^2, where Q_Y is an orthonormal
    basis for the columns of Y."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    Qy, _ = qr(Y, mode="economic")
    num = np.linalg.norm(Qy.T @ X, "fro") ** 2
    den = np.linalg.norm(X, "fro") ** 2
    if den < 1e-12:
        return 0.0
    return float(num / den)


def cca_mean_correlation(X, Y):
    """Mean CCA correlation rho-bar (Eq. 8): mean of the canonical
    correlations between X and Y, via QR + SVD as in Appendix C.2."""
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
    """Convenience wrapper computing all four indices at once."""
    return {
        "linear_cka": linear_cka(X, Y),
        "rbf_cka": rbf_cka(X, Y, rbf_sigma_frac),
        "linear_reg_r2": linear_regression_r2(X, Y),
        "cca_mean_corr": cca_mean_correlation(X, Y),
    }


if __name__ == "__main__":
    # Sanity checks against known properties of CKA:
    rng = np.random.default_rng(0)
    n, d = 50, 8

    X = rng.normal(size=(n, d))

    print("=== Sanity checks ===")
    # 1. CKA(X, X) == 1
    print(f"linear_cka(X, X)  = {linear_cka(X, X):.6f}  (expect ~1.0)")
    print(f"rbf_cka(X, X)     = {rbf_cka(X, X):.6f}  (expect ~1.0)")

    # 2. CKA invariant to orthogonal transform: CKA(X, XQ) == CKA(X, X)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    print(f"linear_cka(X, XQ) = {linear_cka(X, X @ Q):.6f}  (expect ~1.0, orthogonal invariance)")

    # 3. CKA invariant to isotropic scaling: CKA(X, 5X) == CKA(X, X)
    print(f"linear_cka(X, 5X) = {linear_cka(X, 5 * X):.6f}  (expect ~1.0, isotropic scale invariance)")

    # 4. CKA NOT invariant to non-isotropic scaling
    D = np.diag(rng.uniform(0.1, 5, size=d))
    print(f"linear_cka(X, XD) = {linear_cka(X, X @ D):.6f}  (expect <1.0, non-isotropic scaling)")

    # 5. CKA between independent random matrices should be low
    Y_indep = rng.normal(size=(n, d))
    print(f"linear_cka(X, Y_indep) = {linear_cka(X, Y_indep):.6f}  (expect low, near 0)")

    print(f"\nrho_bar_cca(X, X) = {cca_mean_correlation(X, X):.6f}  (expect ~1.0)")
    print(f"R2(X, X) = {linear_regression_r2(X, X):.6f}  (expect ~1.0)")
