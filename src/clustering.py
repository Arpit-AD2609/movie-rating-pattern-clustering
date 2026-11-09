"""GMM (EM) movie clustering, K-Means baseline, and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture

from src.data_loader import RATING_LEVELS
from src.preprocessing import FEATURE_COLS


@dataclass
class FitRecord:
    algorithm: str
    k: int
    labels: np.ndarray
    inertia_or_ll: float
    aic: float | None = None
    bic: float | None = None
    converged: bool | None = None
    n_iter: int | None = None
    silhouette: float | None = None
    sizes: list[int] = field(default_factory=list)
    proba: np.ndarray | None = None
    model: object | None = None


def fit_gmm(
    X: np.ndarray,
    k: int,
    seed: int = 42,
    n_init: int = 3,
    covariance_type: str = "diag",
    max_iter: int = 200,
    reg_covar: float = 1e-6,
) -> FitRecord:
    model = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        n_init=n_init,
        random_state=seed,
        max_iter=max_iter,
        reg_covar=reg_covar,
        init_params="kmeans",
    )
    labels = model.fit_predict(X)
    sizes = np.bincount(labels, minlength=k).tolist()
    return FitRecord(
        algorithm="GMM-EM",
        k=k,
        labels=labels,
        inertia_or_ll=float(model.lower_bound_),
        aic=float(model.aic(X)),
        bic=float(model.bic(X)),
        converged=bool(model.converged_),
        n_iter=int(model.n_iter_),
        sizes=sizes,
        proba=model.predict_proba(X),
        model=model,
    )


def fit_kmeans(X: np.ndarray, k: int, seed: int = 42, n_init: int = 10) -> FitRecord:
    model = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
    labels = model.fit_predict(X)
    sizes = np.bincount(labels, minlength=k).tolist()
    return FitRecord(
        algorithm="KMeans",
        k=k,
        labels=labels,
        inertia_or_ll=float(model.inertia_),
        sizes=sizes,
        model=model,
    )


def silhouette_on_sample(
    X: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
    max_samples: int = 1500,
) -> float | None:
    n = X.shape[0]
    unique = np.unique(labels)
    if len(unique) < 2 or n < 10:
        return None
    if n <= max_samples:
        return float(silhouette_score(X, labels, metric="euclidean"))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_samples, replace=False)
    sampled_labels = labels[idx]
    if len(np.unique(sampled_labels)) < 2:
        return None
    return float(silhouette_score(X[idx], sampled_labels, metric="euclidean"))


def pca_coords(X: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_


def attach_assignments(
    movies: pd.DataFrame,
    gmm: FitRecord,
    kmeans: FitRecord,
    coords: np.ndarray,
) -> pd.DataFrame:
    out = movies.copy()
    out["cluster"] = gmm.labels.astype(int)
    out["kmeans_cluster"] = kmeans.labels.astype(int)
    out["max_responsibility"] = gmm.proba.max(axis=1)
    for j in range(gmm.k):
        out[f"resp_c{j}"] = gmm.proba[:, j]
    out["pca_1"] = coords[:, 0]
    out["pca_2"] = coords[:, 1]
    return out


def cluster_summary(assigned: pd.DataFrame, feature_df: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for cid, grp in assigned.groupby("cluster"):
        profile = grp[FEATURE_COLS].mean()
        rows.append(
            {
                "cluster": int(cid),
                "n_movies": int(len(grp)),
                "share": float(len(grp) / len(assigned)),
                "mean_rating": float(grp["mean_rating"].mean()),
                "mean_std_rating": float(grp["std_rating"].mean()),
                "mean_n_ratings": float(grp["n_ratings"].mean()),
                "mean_max_responsibility": float(grp["max_responsibility"].mean()),
                **{col: float(profile[col]) for col in FEATURE_COLS},
            }
        )
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def representative_movies(assigned: pd.DataFrame, per_cluster: int = 8) -> pd.DataFrame:
    parts = []
    for cid, grp in assigned.groupby("cluster"):
        top = grp.sort_values(
            ["max_responsibility", "n_ratings"],
            ascending=[False, False],
        ).head(per_cluster)
        keep = top[["cluster", "movieId", "title", "genres", "n_ratings", "mean_rating", "std_rating", "max_responsibility"]]
        parts.append(keep)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def compare_k_range(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
    seed: int = 42,
    n_init: int = 3,
    sil_max_samples: int = 1500,
) -> pd.DataFrame:
    rows = []
    for k in range(k_min, k_max + 1):
        gmm = fit_gmm(X, k=k, seed=seed, n_init=n_init)
        km = fit_kmeans(X, k=k, seed=seed)
        gmm_sil = silhouette_on_sample(X, gmm.labels, seed=seed, max_samples=sil_max_samples)
        km_sil = silhouette_on_sample(X, km.labels, seed=seed, max_samples=sil_max_samples)
        rows.append(
            {
                "k": k,
                "gmm_log_likelihood": gmm.inertia_or_ll,
                "gmm_aic": gmm.aic,
                "gmm_bic": gmm.bic,
                "gmm_converged": gmm.converged,
                "gmm_n_iter": gmm.n_iter,
                "gmm_silhouette": gmm_sil,
                "gmm_sizes": gmm.sizes,
                "kmeans_inertia": km.inertia_or_ll,
                "kmeans_silhouette": km_sil,
                "kmeans_sizes": km.sizes,
            }
        )
    return pd.DataFrame(rows)


def recommend_k(comparison: pd.DataFrame) -> dict:
    """Choose K using BIC (primary) with AIC and silhouette as context."""
    if comparison.empty:
        return {"k": None, "reason": "No comparison rows."}
    bic_k = int(comparison.loc[comparison["gmm_bic"].idxmin(), "k"])
    aic_k = int(comparison.loc[comparison["gmm_aic"].idxmin(), "k"])
    sil_col = comparison["gmm_silhouette"]
    sil_k = int(comparison.loc[sil_col.idxmax(), "k"]) if sil_col.notna().any() else None
    reason = (
        f"BIC is minimised at K={bic_k}. AIC is minimised at K={aic_k}. "
        f"GMM silhouette is highest at K={sil_k}. "
        "BIC penalises extra components more strongly than AIC and is used as the primary selection rule "
        "for this GMM. Silhouette measures geometric separation of hard assignments, not mixture likelihood, "
        "so it can disagree with AIC/BIC on overlapping clusters."
    )
    return {"k": bic_k, "aic_k" : aic_k, "silhouette_k": sil_k, "reason": reason}


def manual_silhouette_example(
    X: np.ndarray,
    labels: np.ndarray,
    index: int,
) -> dict:
    """Exact silhouette identity for one movie using Euclidean distance."""
    x = X[index]
    own = labels[index]
    same = labels == own
    others_same = same.copy()
    others_same[index] = False
    if others_same.sum() == 0:
        a = 0.0
    else:
        a = float(np.mean(np.linalg.norm(X[others_same] - x, axis=1)))
    b = np.inf
    nearest = None
    for cid in np.unique(labels):
        if cid == own:
            continue
        d = float(np.mean(np.linalg.norm(X[labels == cid] - x, axis=1)))
        if d < b:
            b = d
            nearest = int(cid)
    denom = max(a, b)
    s = float((b - a) / denom) if denom > 0 else 0.0
    sklearn_s = float(silhouette_samples(X, labels, metric="euclidean")[index])
    return {
        "index": int(index),
        "cluster": int(own),
        "nearest_other_cluster": nearest,
        "a": a,
        "b": float(b),
        "s": s,
        "sklearn_s": sklearn_s,
        "matches_sklearn": bool(np.isclose(s, sklearn_s, atol=1e-8)),
    }


def rating_profile_table(assigned: pd.DataFrame) -> pd.DataFrame:
    means = assigned.groupby("cluster")[FEATURE_COLS].mean()
    means.columns = [str(r) for r in RATING_LEVELS]
    return means.reset_index()
