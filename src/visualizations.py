"""Plotly charts for the dashboard and report exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import RATING_LEVELS
from src.presentation import (
    ACCENT,
    GMM_COLOR,
    GRID,
    INK,
    KMEANS_COLOR,
    MUTED,
    PAPER,
    SURFACE,
    SURFACE_2,
    cluster_color,
    cluster_label,
    color_map_for,
)
from src.preprocessing import FEATURE_COLS

HOVER = SURFACE_2

DARK_LAYOUT = dict(
    paper_bgcolor=PAPER,
    plot_bgcolor=SURFACE,
    font=dict(color=INK, size=13, family="Source Sans 3, sans-serif"),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        font=dict(size=12, color=INK),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    margin=dict(l=48, r=24, t=64, b=48),
    hoverlabel=dict(bgcolor=HOVER, font_size=12, font_color=INK, bordercolor=GRID),
    title=dict(font=dict(size=16, color=INK), x=0, xanchor="left"),
)


def _axes(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor=GRID,
        tickfont=dict(color=MUTED, size=12),
        title_font=dict(color=MUTED, size=12),
        showgrid=True,
        gridwidth=0.6,
    )
    fig.update_yaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor=GRID,
        tickfont=dict(color=MUTED, size=12),
        title_font=dict(color=MUTED, size=12),
        showgrid=True,
        gridwidth=0.6,
    )
    return fig


def apply_chart_theme(fig: go.Figure, height: int = 380, showlegend: bool | None = None) -> go.Figure:
    fig.update_layout(height=height, **DARK_LAYOUT)
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    return _axes(fig)


def _cluster_labels(series: pd.Series) -> pd.Series:
    return series.map(cluster_label)


def _discrete_map(assigned_or_ids) -> dict[str, str]:
    if hasattr(assigned_or_ids, "unique"):
        ids = assigned_or_ids.unique()
    else:
        ids = assigned_or_ids
    return color_map_for(ids)


def rating_frequency(ratings: pd.DataFrame) -> go.Figure:
    counts = ratings["rating"].value_counts().sort_index()
    fig = px.bar(
        x=[f"{v:g}" for v in counts.index],
        y=counts.values,
        labels={"x": "Star rating", "y": "Number of ratings"},
        title="How often each star rating appears",
        color_discrete_sequence=[ACCENT],
    )
    fig.update_traces(
        hovertemplate="Rating %{x}<br>%{y:,} ratings<extra></extra>",
        marker_line_width=0,
    )
    return apply_chart_theme(fig, showlegend=False)


def ratings_per_movie(counts: pd.Series) -> go.Figure:
    fig = px.histogram(
        x=counts,
        nbins=40,
        labels={"x": "Ratings per movie", "y": "Movies"},
        title="How many ratings each movie received",
        color_discrete_sequence=[GMM_COLOR],
    )
    fig.update_traces(hovertemplate="%{x:.0f} ratings<br>%{y} movies<extra></extra>")
    return apply_chart_theme(fig, showlegend=False)


def mean_rating_distribution(means: pd.Series) -> go.Figure:
    fig = px.histogram(
        x=means,
        nbins=30,
        labels={"x": "Mean movie rating", "y": "Movies"},
        title="Distribution of average movie ratings",
        color_discrete_sequence=["#6FD6FF"],
    )
    fig.update_traces(hovertemplate="Mean %{x:.2f}<br>%{y} movies<extra></extra>")
    return apply_chart_theme(fig, showlegend=False)


def movie_rating_histogram(ratings: pd.DataFrame, title: str) -> go.Figure:
    counts = ratings["rating"].value_counts().sort_index()
    fig = px.bar(
        x=[f"{v:g}" for v in counts.index],
        y=counts.values,
        labels={"x": "Star rating", "y": "Count"},
        title=title,
        color_discrete_sequence=[ACCENT],
    )
    fig.update_traces(hovertemplate="Rating %{x}<br>%{y} ratings<extra></extra>", marker_line_width=0)
    return apply_chart_theme(fig, height=320, showlegend=False)


def cluster_size_bar(summary: pd.DataFrame) -> go.Figure:
    df = summary.copy()
    df["label"] = _cluster_labels(df["cluster"])
    fig = px.bar(
        df,
        x="label",
        y="n_movies",
        color="label",
        color_discrete_map=_discrete_map(df["cluster"]),
        labels={"label": "Cluster", "n_movies": "Movies"},
        title="Movies assigned to each cluster",
        text="n_movies",
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate="%{x}<br>%{y} movies<extra></extra>",
        marker_line_width=0,
        cliponaxis=False,
    )
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode="hide")
    return apply_chart_theme(fig, height=400, showlegend=False)


def pca_scatter(assigned: pd.DataFrame) -> go.Figure:
    df = assigned.copy()
    df["label"] = _cluster_labels(df["cluster"])
    fig = px.scatter(
        df,
        x="pca_1",
        y="pca_2",
        color="label",
        color_discrete_map=_discrete_map(df["cluster"]),
        hover_data={
            "title": True,
            "mean_rating": ":.2f",
            "n_ratings": True,
            "max_responsibility": ":.3f",
            "pca_1": ":.3f",
            "pca_2": ":.3f",
            "label": False,
        },
        title="2D PCA of rating-share features (not the clustering space)",
        opacity=0.78,
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0)))
    fig.update_xaxes(title="PCA 1")
    fig.update_yaxes(title="PCA 2")
    return apply_chart_theme(fig, height=480)


def pca_scatter_with_ellipses(assigned: pd.DataFrame) -> go.Figure:
    df = assigned.copy()
    df["label"] = _cluster_labels(df["cluster"])
    fig = px.scatter(
        df,
        x="pca_1",
        y="pca_2",
        color="label",
        color_discrete_map=_discrete_map(df["cluster"]),
        hover_data={
            "title": True,
            "mean_rating": ":.2f",
            "n_ratings": True,
            "max_responsibility": ":.3f",
            "label": False,
        },
        title="PCA projection with 95% spread ellipses in 2D",
        opacity=0.62,
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0)))
    clusters = sorted(assigned["cluster"].unique().tolist())
    for cid in clusters:
        sub = assigned[assigned["cluster"] == cid]
        if len(sub) < 3:
            continue
        pts = sub[["pca_1", "pca_2"]].to_numpy(dtype=float)
        centre = pts.mean(axis=0)
        cov = np.cov(pts, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov), 0.0], [0.0, float(cov)]])
        det = np.linalg.det(cov)
        if not np.isfinite(det) or det <= 1e-18:
            continue
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-12, None)
        scale = np.sqrt(5.991)
        theta = np.linspace(0, 2 * np.pi, 120)
        ellipse = np.column_stack((np.cos(theta), np.sin(theta))) @ np.diag(np.sqrt(vals) * scale) @ vecs.T
        ellipse = ellipse + centre
        fig.add_trace(
            go.Scatter(
                x=ellipse[:, 0],
                y=ellipse[:, 1],
                mode="lines",
                name=f"{cluster_label(cid)} ellipse",
                line=dict(color=cluster_color(cid), width=1.8),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.update_xaxes(title="PCA 1")
    fig.update_yaxes(title="PCA 2")
    return apply_chart_theme(fig, height=520)


def cluster_profiles(assigned: pd.DataFrame) -> go.Figure:
    means = assigned.groupby("cluster")[FEATURE_COLS].mean()
    fig = go.Figure()
    x = [str(r) for r in RATING_LEVELS]
    for cid, row in means.iterrows():
        fig.add_trace(
            go.Scatter(
                x=x,
                y=row.values,
                mode="lines+markers",
                name=cluster_label(cid),
                line=dict(color=cluster_color(cid), width=2.2),
                marker=dict(size=7),
                hovertemplate="%{fullData.name}<br>%{x} stars: %{y:.1%}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Mean rating-share profile by cluster",
        xaxis_title="Star rating",
        yaxis_title="Mean share of observed ratings",
        yaxis_tickformat=".0%",
    )
    return apply_chart_theme(fig, height=430)


def rating_profile_heatmap(assigned: pd.DataFrame) -> go.Figure:
    profile = assigned.groupby("cluster")[FEATURE_COLS].mean()
    y_labels = [cluster_label(cid) for cid in profile.index]
    x_labels = [str(r) for r in RATING_LEVELS]
    fig = px.imshow(
        profile.to_numpy(),
        x=x_labels,
        y=y_labels,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=["#191B20", "#3E4A24", "#8BA43D", "#C7F36B"],
        title="Mean rating-share heatmap",
        labels={"color": "Share"},
    )
    fig.update_traces(hovertemplate="%{y}<br>%{x} stars: %{z:.1%}<extra></extra>")
    fig.update_xaxes(title="Star rating", side="bottom")
    fig.update_yaxes(title="")
    fig.update_layout(coloraxis_colorbar=dict(tickformat=".0%"))
    return apply_chart_theme(fig, height=380, showlegend=False)


def aic_bic_plot(comparison: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=comparison["k"],
            y=comparison["gmm_aic"],
            mode="lines+markers",
            name="AIC (GMM only)",
            line=dict(color=GMM_COLOR, width=2),
            hovertemplate="K=%{x}<br>AIC=%{y:,.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=comparison["k"],
            y=comparison["gmm_bic"],
            mode="lines+markers",
            name="BIC (GMM only)",
            line=dict(color=ACCENT, width=2),
            hovertemplate="K=%{x}<br>BIC=%{y:,.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="AIC and BIC versus number of GMM components",
        xaxis_title="K",
        yaxis_title="Information criterion (lower is better)",
        xaxis=dict(dtick=1),
    )
    return apply_chart_theme(fig)


def silhouette_compare(comparison: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=comparison["k"],
            y=comparison["gmm_silhouette"],
            name="GMM (hard labels)",
            marker_color=GMM_COLOR,
            hovertemplate="K=%{x}<br>GMM silhouette=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=comparison["k"],
            y=comparison["kmeans_silhouette"],
            name="K-Means",
            marker_color=KMEANS_COLOR,
            hovertemplate="K=%{x}<br>K-Means silhouette=%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        barmode="group",
        title="Silhouette of hard assignments versus K",
        xaxis_title="K",
        yaxis_title="Silhouette (higher is more separated)",
        xaxis=dict(dtick=1),
    )
    fig.add_hline(y=0, line_dash="dot", line_color=MUTED, line_width=1)
    return apply_chart_theme(fig)


def gmm_vs_kmeans_sizes(gmm_sizes: list[int], km_sizes: list[int]) -> go.Figure:
    k = [cluster_label(i) for i in range(len(gmm_sizes))]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=k,
            y=gmm_sizes,
            name="GMM",
            marker_color=GMM_COLOR,
            hovertemplate="%{x}<br>GMM: %{y} movies<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=k,
            y=km_sizes,
            name="K-Means",
            marker_color=KMEANS_COLOR,
            hovertemplate="%{x}<br>K-Means: %{y} movies<extra></extra>",
        )
    )
    fig.update_layout(
        barmode="group",
        title="Cluster sizes at the selected K: GMM versus K-Means",
        xaxis_title="Component / cluster index",
        yaxis_title="Movies",
    )
    return apply_chart_theme(fig, height=400)


def em_ll_curve(log_likelihoods: list[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(log_likelihoods) + 1)),
            y=log_likelihoods,
            mode="lines+markers",
            line=dict(color=ACCENT, width=2),
            marker=dict(size=7),
            hovertemplate="Iteration %{x}<br>Mean log-likelihood %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="From-scratch EM log-likelihood (synthetic 1-D data)",
        xaxis_title="Iteration",
        yaxis_title="Mean log-likelihood",
    )
    return apply_chart_theme(fig, height=320, showlegend=False)


def distance_to_mean_box(assigned: pd.DataFrame, distances: np.ndarray) -> go.Figure:
    df = assigned[["cluster"]].copy()
    df["label"] = _cluster_labels(df["cluster"])
    df["distance"] = distances
    fig = px.box(
        df,
        x="label",
        y="distance",
        color="label",
        color_discrete_map=_discrete_map(df["cluster"]),
        labels={"label": "Cluster", "distance": "Euclidean distance"},
        title="Distance to assigned GMM component mean (10-D rating shares)",
    )
    fig.update_traces(showlegend=False)
    return apply_chart_theme(fig, height=400, showlegend=False)


def silhouette_by_cluster_box(cluster_ids: np.ndarray, silhouettes: np.ndarray) -> go.Figure:
    df = pd.DataFrame({"cluster": cluster_ids, "silhouette": silhouettes})
    df["label"] = _cluster_labels(df["cluster"])
    fig = px.box(
        df,
        x="label",
        y="silhouette",
        color="label",
        color_discrete_map=_discrete_map(df["cluster"]),
        labels={"label": "Cluster", "silhouette": "Silhouette"},
        title="Per-cluster silhouette of hard GMM assignments",
    )
    fig.add_hline(y=0, line_dash="dot", line_color=MUTED)
    fig.update_traces(showlegend=False)
    return apply_chart_theme(fig, height=400, showlegend=False)


def max_responsibility_box(assigned: pd.DataFrame) -> go.Figure:
    df = assigned[["cluster", "max_responsibility"]].copy()
    df["label"] = _cluster_labels(df["cluster"])
    fig = px.box(
        df,
        x="label",
        y="max_responsibility",
        color="label",
        color_discrete_map=_discrete_map(df["cluster"]),
        labels={"label": "Cluster", "max_responsibility": "Max posterior"},
        title="Maximum component responsibility by cluster",
    )
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    fig.update_traces(showlegend=False)
    return apply_chart_theme(fig, height=380, showlegend=False)


def gmm_means_heatmap(means: np.ndarray) -> go.Figure:
    y_labels = [cluster_label(i) for i in range(means.shape[0])]
    x_labels = [str(r) for r in RATING_LEVELS]
    fig = px.imshow(
        means,
        x=x_labels,
        y=y_labels,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=["#191B20", "#3E4A24", "#8BA43D", "#C7F36B"],
        title="Fitted GMM component means (rating-share space)",
        labels={"color": "Mean share"},
    )
    fig.update_traces(hovertemplate="%{y}<br>%{x} stars: %{z:.3f}<extra></extra>")
    fig.update_xaxes(title="Star rating")
    fig.update_yaxes(title="")
    return apply_chart_theme(fig, height=360, showlegend=False)


def save_plotly_png(fig: go.Figure, path: Path) -> None:
    """Best-effort static export. HTML is always written as a fallback."""
    path.parent.mkdir(parents=True, exist_ok=True)
    html_path = path.with_suffix(".html")
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    try:
        fig.write_image(str(path))
    except Exception:
        pass
