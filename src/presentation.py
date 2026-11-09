"""Shared presentation helpers: cluster colours, labels, and profile copy.

These functions summarise values already computed by the analysis pipeline.
They do not invent metrics or movie characteristics.
"""

from __future__ import annotations

import pandas as pd

from src.data_loader import RATING_LEVELS
from src.preprocessing import FEATURE_COLS

CLUSTER_COLORS = [
    "#C7F36B",
    "#8AA9FF",
    "#E8A87C",
    "#6FD6FF",
    "#C792EA",
    "#E6D37A",
    "#7BD389",
    "#FFB86C",
]

ACCENT = "#C7F36B"
MUTED = "#A5A8B0"
INK = "#F3F0E8"
PAPER = "#101114"
SURFACE = "#191B20"
SURFACE_2 = "#22252B"
GRID = "#2C3038"
GMM_COLOR = "#8AA9FF"
KMEANS_COLOR = "#E8A87C"


def cluster_color(cluster_id: int) -> str:
    return CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)]


def cluster_label(cluster_id: int) -> str:
    return f"Cluster {int(cluster_id)}"


def color_map_for(cluster_ids) -> dict[str, str]:
    return {cluster_label(cid): cluster_color(cid) for cid in cluster_ids}


def feature_to_stars(col: str) -> str:
    idx = FEATURE_COLS.index(col) if col in FEATURE_COLS else None
    if idx is None:
        return col
    value = RATING_LEVELS[idx]
    return f"{value:g}"


def dominant_rating_shares(row: pd.Series, top_n: int = 2) -> list[tuple[str, float]]:
    ranked = sorted(
        ((feature_to_stars(col), float(row[col])) for col in FEATURE_COLS if col in row.index),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[:top_n]


def interpret_cluster_row(row: pd.Series) -> str:
    """Plain-language sketch of a cluster from its mean rating-share profile."""
    mean_r = float(row["mean_rating"])
    n = int(row["n_movies"])
    n_ratings = float(row["mean_n_ratings"])
    certainty = float(row["mean_max_responsibility"])
    top = dominant_rating_shares(row, top_n=2)
    peak = ", ".join(f"{stars}-star ({share:.0%})" for stars, share in top)

    if mean_r < 3.0:
        tone = "lower-rated on average"
    elif mean_r < 3.5:
        tone = "mid-range on average"
    elif mean_r < 3.8:
        tone = "generally well regarded"
    else:
        tone = "among the higher-rated groups"

    return (
        f"{n} movies, {tone} (mean {mean_r:.2f}). "
        f"Typical observed histogram peaks at {peak}. "
        f"Average {n_ratings:.0f} ratings per title; "
        f"mean MAP responsibility {certainty:.2f}."
    )


def cluster_briefs(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["label"] = out["cluster"].map(cluster_label)
    out["reading"] = out.apply(interpret_cluster_row, axis=1)
    return out


def fmt_pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def fmt_bool(flag: bool | None) -> str:
    if flag is None:
        return "—"
    return "Yes" if flag else "No"
