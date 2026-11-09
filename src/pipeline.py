"""Single analysis pipeline shared by the Streamlit app, notebook, and CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.clustering import (
    attach_assignments,
    cluster_summary,
    compare_k_range,
    fit_gmm,
    fit_kmeans,
    manual_silhouette_example,
    pca_coords,
    recommend_k,
    representative_movies,
    silhouette_on_sample,
)
from src.data_loader import ROOT, dataset_overview, load_raw
from src.em_from_scratch import run_demo
from src.preprocessing import FEATURE_COLS, feature_matrix, movie_rating_features, movies_below_threshold
from src.visualizations import (
    aic_bic_plot,
    cluster_profiles,
    cluster_size_bar,
    em_ll_curve,
    gmm_vs_kmeans_sizes,
    mean_rating_distribution,
    pca_scatter,
    rating_frequency,
    ratings_per_movie,
    save_plotly_png,
    silhouette_compare,
)

OUTPUT = ROOT / "output"
FIG_DIR = OUTPUT / "figures"
TABLE_DIR = OUTPUT / "tables"
RESULT_DIR = OUTPUT / "results"

DEFAULT_MIN_RATINGS = 20
DEFAULT_K = 7
DEFAULT_SEED = 42
K_MIN, K_MAX = 2, 8
GMM_N_INIT = 3
SIL_MAX_SAMPLES = 1500


def run_analysis(
    min_ratings: int = DEFAULT_MIN_RATINGS,
    k: int = DEFAULT_K,
    seed: int = DEFAULT_SEED,
    data: dict[str, pd.DataFrame] | None = None,
) -> dict:
    data = data if data is not None else load_raw()
    overview = dataset_overview(data)
    movies_feat, prep_stats = movie_rating_features(
        data["ratings"], data["movies"], min_ratings=min_ratings
    )
    excluded = movies_below_threshold(data["ratings"], data["movies"], min_ratings=min_ratings)
    if len(movies_feat) < k:
        raise ValueError(
            f"Only {len(movies_feat)} movies remain after min_ratings={min_ratings}; "
            f"need at least K={k}."
        )

    X = feature_matrix(movies_feat)
    comparison = compare_k_range(
        X, k_min=K_MIN, k_max=K_MAX, seed=seed, n_init=GMM_N_INIT, sil_max_samples=SIL_MAX_SAMPLES
    )
    rec = recommend_k(comparison)

    gmm = fit_gmm(X, k=k, seed=seed, n_init=GMM_N_INIT)
    km = fit_kmeans(X, k=k, seed=seed)
    gmm.silhouette = silhouette_on_sample(X, gmm.labels, seed=seed, max_samples=SIL_MAX_SAMPLES)
    km.silhouette = silhouette_on_sample(X, km.labels, seed=seed, max_samples=SIL_MAX_SAMPLES)

    coords, pca_var = pca_coords(X, seed=seed)
    assigned = attach_assignments(movies_feat, gmm, km, coords)
    summary = cluster_summary(assigned)
    reps = representative_movies(assigned)

    rng = np.random.default_rng(seed)
    example_idx = int(rng.integers(0, len(assigned)))
    manual_s = manual_silhouette_example(X, gmm.labels, example_idx)
    manual_s["title"] = str(assigned.iloc[example_idx]["title"])
    manual_s["movieId"] = int(assigned.iloc[example_idx]["movieId"])

    em_demo = run_demo(seed=seed)

    per_movie_counts = data["ratings"].groupby("movieId").size()
    per_movie_means = data["ratings"].groupby("movieId")["rating"].mean()

    return {
        "config": {
            "min_ratings": min_ratings,
            "k": k,
            "seed": seed,
            "n_init": GMM_N_INIT,
            "covariance_type": "diag",
            "k_tune_range": [K_MIN, K_MAX],
            "silhouette_max_samples": SIL_MAX_SAMPLES,
        },
        "data": data,
        "overview": overview,
        "prep_stats": prep_stats,
        "excluded_movies": excluded,
        "movies_feat": movies_feat,
        "X": X,
        "comparison": comparison,
        "recommendation": rec,
        "gmm": gmm,
        "kmeans": km,
        "assigned": assigned,
        "summary": summary,
        "representatives": reps,
        "pca_var": pca_var,
        "manual_silhouette": manual_s,
        "em_demo": em_demo,
        "per_movie_counts": per_movie_counts,
        "per_movie_means": per_movie_means,
        "feature_cols": FEATURE_COLS,
    }


def persist_outputs(result: dict) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    assigned: pd.DataFrame = result["assigned"]
    summary: pd.DataFrame = result["summary"]
    comparison: pd.DataFrame = result["comparison"]
    reps: pd.DataFrame = result["representatives"]
    ratings = result["data"]["ratings"]

    export_cols = [
        "movieId",
        "title",
        "genres",
        "n_ratings",
        "mean_rating",
        "std_rating",
        "cluster",
        "kmeans_cluster",
        "max_responsibility",
        "pca_1",
        "pca_2",
    ] + [c for c in assigned.columns if c.startswith("resp_c")]
    assigned[export_cols].to_csv(RESULT_DIR / "movie_cluster_assignments.csv", index=False)
    summary.to_csv(TABLE_DIR / "cluster_summary.csv", index=False)
    comparison.to_csv(TABLE_DIR / "model_comparison.csv", index=False)
    reps.to_csv(TABLE_DIR / "representative_movies.csv", index=False)
    pd.DataFrame([result["manual_silhouette"]]).to_csv(TABLE_DIR / "manual_silhouette.csv", index=False)
    meta = {**result["config"], **{k: v for k, v in result["prep_stats"].items() if not isinstance(v, (list, dict))}}
    pd.DataFrame([meta]).to_csv(TABLE_DIR / "run_metadata.csv", index=False)

    figs = {
        "rating_frequency": rating_frequency(ratings),
        "ratings_per_movie": ratings_per_movie(result["per_movie_counts"]),
        "mean_rating_distribution": mean_rating_distribution(result["per_movie_means"]),
        "aic_bic": aic_bic_plot(comparison),
        "silhouette_compare": silhouette_compare(comparison),
        "cluster_sizes": cluster_size_bar(summary),
        "pca_scatter": pca_scatter(assigned),
        "cluster_profiles": cluster_profiles(assigned),
        "gmm_vs_kmeans_sizes": gmm_vs_kmeans_sizes(result["gmm"].sizes, result["kmeans"].sizes),
        "em_ll_curve": em_ll_curve(result["em_demo"].log_likelihoods),
    }
    written = {}
    for name, fig in figs.items():
        path = FIG_DIR / f"{name}.png"
        save_plotly_png(fig, path)
        written[name] = str(path.with_suffix(".html"))
    written["assignments_csv"] = str(RESULT_DIR / "movie_cluster_assignments.csv")
    written["comparison_csv"] = str(TABLE_DIR / "model_comparison.csv")
    written["summary_csv"] = str(TABLE_DIR / "cluster_summary.csv")
    return written
