"""Load and inspect the local MovieLens CSV files. Original files are never modified."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RATING_LEVELS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


def raw_paths() -> dict[str, Path]:
    return {
        "ratings": RAW_DIR / "ratings.csv",
        "movies": RAW_DIR / "movies.csv",
        "tags": RAW_DIR / "tags.csv",
        "links": RAW_DIR / "links.csv",
    }


def missing_raw_files() -> list[str]:
    return [name for name, path in raw_paths().items() if not path.exists()]


def load_raw() -> dict[str, pd.DataFrame]:
    missing = missing_raw_files()
    if missing:
        raise FileNotFoundError(
            "Missing dataset file(s): "
            + ", ".join(missing)
            + f". Place the CSVs in {RAW_DIR}."
        )
    paths = raw_paths()
    return {
        "ratings": pd.read_csv(paths["ratings"]),
        "movies": pd.read_csv(paths["movies"]),
        "tags": pd.read_csv(paths["tags"]),
        "links": pd.read_csv(paths["links"]),
    }


def profile_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        series = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def dataset_overview(data: dict[str, pd.DataFrame]) -> dict:
    ratings = data["ratings"]
    movies = data["movies"]
    tags = data["tags"]
    links = data["links"]
    rated_ids = set(ratings["movieId"].unique())
    catalog_ids = set(movies["movieId"].unique())
    return {
        "n_ratings": int(len(ratings)),
        "n_users": int(ratings["userId"].nunique()),
        "n_movies_catalog": int(len(movies)),
        "n_movies_rated": int(ratings["movieId"].nunique()),
        "n_movies_unrated": int(len(catalog_ids - rated_ids)),
        "n_ratings_unmatched": int((~ratings["movieId"].isin(movies["movieId"])).sum()),
        "rating_min": float(ratings["rating"].min()),
        "rating_max": float(ratings["rating"].max()),
        "unique_ratings": [float(x) for x in sorted(ratings["rating"].dropna().unique())],
        "n_tags": int(len(tags)),
        "n_links": int(len(links)),
        "duplicate_ratings": int(ratings.duplicated().sum()),
        "duplicate_movies": int(movies.duplicated().sum()),
        "missing_ratings": {k: int(v) for k, v in ratings.isna().sum().items()},
        "missing_movies": {k: int(v) for k, v in movies.isna().sum().items()},
    }
