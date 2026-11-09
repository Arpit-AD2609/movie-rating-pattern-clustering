"""Build per-movie rating-distribution features. Unobserved ratings are never treated as zeros."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import RATING_LEVELS


FEATURE_COLS = [f"p_{str(r).replace('.', '_')}" for r in RATING_LEVELS]


def _valid_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    valid = ratings.dropna(subset=["movieId", "rating"]).copy()
    valid["movieId"] = valid["movieId"].astype(int)
    valid = valid[valid["rating"].isin(RATING_LEVELS)]
    return valid


def movie_rating_features(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    min_ratings: int = 20,
) -> tuple[pd.DataFrame, dict]:
    """Return one row per movie that meets the minimum-rating threshold.

    Features are empirical proportions of observed ratings at each half-star
    level. Missing user–movie pairs are omitted; they are not imputed.
    """
    valid = _valid_ratings(ratings)
    n_invalid = int(len(ratings) - len(valid))

    counts = (
        valid.groupby(["movieId", "rating"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=RATING_LEVELS, fill_value=0)
    )
    n_obs = counts.sum(axis=1)
    eligible = counts.loc[n_obs >= min_ratings].copy()
    n_obs_elig = n_obs.loc[eligible.index]
    props = eligible.div(n_obs_elig, axis=0)
    props.columns = FEATURE_COLS

    mean_rating = valid.groupby("movieId")["rating"].mean()
    std_rating = valid.groupby("movieId")["rating"].std(ddof=0)
    features = props.copy()
    features["n_ratings"] = n_obs_elig.astype(int)
    features["mean_rating"] = mean_rating.loc[features.index]
    features["std_rating"] = std_rating.loc[features.index].fillna(0.0)
    features = features.reset_index()

    movie_meta = movies[["movieId", "title", "genres"]].drop_duplicates("movieId")
    merged = features.merge(movie_meta, on="movieId", how="left")
    unmatched_titles = int(merged["title"].isna().sum())
    merged["title"] = merged["title"].fillna(f"(title missing)")
    merged["genres"] = merged["genres"].fillna("(genres missing)")

    row_sums = merged[FEATURE_COLS].sum(axis=1)
    stats = {
        "min_ratings": int(min_ratings),
        "n_rating_rows_input": int(len(ratings)),
        "n_invalid_or_dropped_ratings": n_invalid,
        "n_movies_with_any_valid_rating": int(n_obs.shape[0]),
        "n_movies_retained": int(len(merged)),
        "n_movies_excluded": int(n_obs.shape[0] - len(merged)),
        "unmatched_titles": unmatched_titles,
        "feature_row_sum_min": float(row_sums.min()) if len(merged) else None,
        "feature_row_sum_max": float(row_sums.max()) if len(merged) else None,
        "n_feature_dims": len(FEATURE_COLS),
        "rating_levels": list(RATING_LEVELS),
    }
    return merged, stats


def movies_below_threshold(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    min_ratings: int = 20,
    limit: int | None = None,
) -> pd.DataFrame:
    """Catalog movies that have at least one valid rating but fewer than min_ratings.

    Unrated catalog titles are not included here; they never enter the feature table.
    """
    valid = _valid_ratings(ratings)
    n_obs = valid.groupby("movieId").size().rename("n_ratings")
    below = n_obs[n_obs < min_ratings].reset_index()
    meta = movies[["movieId", "title", "genres"]].drop_duplicates("movieId")
    out = below.merge(meta, on="movieId", how="left")
    means = valid.groupby("movieId")["rating"].mean().rename("mean_rating")
    out = out.merge(means, on="movieId", how="left")
    out = out.sort_values(["n_ratings", "title"], ascending=[True, True])
    if limit is not None:
        out = out.head(limit)
    return out.reset_index(drop=True)


def feature_matrix(movie_df: pd.DataFrame) -> np.ndarray:
    X = movie_df[FEATURE_COLS].to_numpy(dtype=float)
    return X
