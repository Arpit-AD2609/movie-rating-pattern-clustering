"""One-dimensional two-component Gaussian mixture estimated with EM.

Educational only. The MovieLens analysis uses scikit-learn GaussianMixture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EM1DResult:
    weights: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    log_likelihoods: list[float] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False
    responsibilities: np.ndarray | None = None
    X: np.ndarray | None = None


def _logsumexp(a: np.ndarray, axis: int = 1) -> np.ndarray:
    peak = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(peak, axis=axis) + np.log(np.sum(np.exp(a - peak), axis=axis))


def _log_gaussian(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    var = max(float(var), 1e-8)
    return -0.5 * (np.log(2.0 * np.pi * var) + (x - mean) ** 2 / var)


def e_step(X: np.ndarray, weights: np.ndarray, means: np.ndarray, variances: np.ndarray) -> tuple[np.ndarray, float]:
    """Return responsibilities gamma (n, k) and average log-likelihood."""
    n = X.shape[0]
    k = weights.shape[0]
    log_comp = np.empty((n, k), dtype=float)
    for j in range(k):
        log_comp[:, j] = np.log(max(weights[j], 1e-12)) + _log_gaussian(X, means[j], variances[j])
    log_norm = _logsumexp(log_comp, axis=1)
    resp = np.exp(log_comp - log_norm[:, None])
    return resp, float(np.mean(log_norm))


def m_step(X: np.ndarray, resp: np.ndarray, var_floor: float = 1e-4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = X.shape[0]
    n_k = resp.sum(axis=0) + 1e-12
    weights = n_k / n
    means = (resp.T @ X) / n_k
    sq = (X[:, None] - means[None, :]) ** 2
    variances = np.maximum((resp * sq).sum(axis=0) / n_k, var_floor)
    return weights, means, variances


def fit_em_1d(
    X: np.ndarray,
    n_components: int = 2,
    max_iter: int = 100,
    tol: float = 1e-6,
    seed: int = 42,
    var_floor: float = 1e-4,
) -> EM1DResult:
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float).ravel()
    n = X.shape[0]
    if n < n_components:
        raise ValueError("Need at least as many points as components.")

    qs = np.linspace(10, 90, n_components)
    means = np.percentile(X, qs)
    variances = np.full(n_components, max(float(np.var(X)), var_floor))
    weights = np.full(n_components, 1.0 / n_components)

    ll_hist: list[float] = []
    resp = None
    converged = False
    n_iter = 0
    for i in range(1, max_iter + 1):
        resp, ll = e_step(X, weights, means, variances)
        weights, means, variances = m_step(X, resp, var_floor=var_floor)
        ll_hist.append(ll)
        n_iter = i
        if i > 1 and abs(ll_hist[-1] - ll_hist[-2]) < tol:
            converged = True
            break

    return EM1DResult(
        weights=weights,
        means=means,
        variances=variances,
        log_likelihoods=ll_hist,
        n_iter=n_iter,
        converged=converged,
        responsibilities=resp,
        X=X,
    )


def make_synthetic_1d(n: int = 400, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    x1 = rng.normal(loc=-2.0, scale=0.6, size=n1)
    x2 = rng.normal(loc=2.2, scale=0.7, size=n2)
    X = np.concatenate([x1, x2])
    labels = np.concatenate([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])
    return X, labels


def run_demo(seed: int = 42) -> EM1DResult:
    X, _ = make_synthetic_1d(seed=seed)
    return fit_em_1d(X, n_components=2, seed=seed)
