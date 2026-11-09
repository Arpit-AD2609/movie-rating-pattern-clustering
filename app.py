"""Movie Clustering Analytics Lab — Streamlit app for EM/GMM movie segmentation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import silhouette_samples, silhouette_score

from src.clustering import rating_profile_table
from src.data_loader import RAW_DIR, dataset_overview, load_raw, missing_raw_files, profile_frame
from src.em_from_scratch import run_demo
from src.pipeline import DEFAULT_K, DEFAULT_MIN_RATINGS, DEFAULT_SEED, persist_outputs, run_analysis
from src.presentation import cluster_briefs, cluster_label, fmt_bool, fmt_pct
from src.preprocessing import movie_rating_features, movies_below_threshold
from src.visualizations import (
    aic_bic_plot,
    cluster_profiles,
    cluster_size_bar,
    distance_to_mean_box,
    em_ll_curve,
    gmm_means_heatmap,
    gmm_vs_kmeans_sizes,
    max_responsibility_box,
    mean_rating_distribution,
    movie_rating_histogram,
    pca_scatter,
    pca_scatter_with_ellipses,
    rating_frequency,
    rating_profile_heatmap,
    ratings_per_movie,
    silhouette_by_cluster_box,
    silhouette_compare,
)

st.set_page_config(
    page_title="Movie Clustering Analytics Lab",
    page_icon=":material/movie:",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAV = [
    ("overview", "Overview"),
    ("explorer", "Dataset explorer"),
    ("lab", "EM clustering lab"),
    ("results", "Cluster results"),
    ("distribution", "Cluster distribution"),
    ("compare", "Model comparison"),
    ("method", "Methodology"),
    ("export", "Report and exports"),
]
NAV_LABELS = {key: label for key, label in NAV}

BRAND_CSS = """
<style>
  h1, h2, h3 { letter-spacing: -0.03em; }
  [data-testid="stSidebar"] { min-width: 17.5rem; }
  [data-testid="stSidebar"] .stRadio > label { font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: #A5A8B0; }
  [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    padding: 0.35rem 0.45rem;
    border-radius: 0.3rem;
  }
  [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: #22252B;
  }
  .stPlotlyChart, [data-testid="stDataFrame"] {
    border: 1px solid #2C3038;
    background: #191B20;
  }
  [data-testid="stMetric"] { background: #191B20; padding: 0.7rem 0.85rem; border: 1px solid #2C3038; }
  button:hover { filter: brightness(1.06); }
</style>
"""


@st.cache_data(show_spinner=False)
def cached_raw() -> dict[str, pd.DataFrame]:
    return load_raw()


@st.cache_data(show_spinner=False)
def cached_feature_preview(min_ratings: int) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    data = cached_raw()
    feats, stats = movie_rating_features(data["ratings"], data["movies"], min_ratings=min_ratings)
    excluded = movies_below_threshold(data["ratings"], data["movies"], min_ratings=min_ratings)
    return feats, stats, excluded


def inject_brand() -> None:
    st.html(BRAND_CSS)


def chart(fig, alt: str, *, height: int | None = None) -> None:
    kwargs = {"width": "stretch", "alt": alt, "theme": None}
    if height is not None:
        kwargs["height"] = height
    st.plotly_chart(fig, **kwargs)


def get_result() -> dict | None:
    return st.session_state.get("result")


def require_result() -> dict | None:
    result = get_result()
    if result is None:
        st.info(
            "Set the minimum-ratings threshold, K, and seed in the sidebar, then choose **Run analysis**.",
            icon=":material/science:",
        )
        return None
    return result


def params_match_result(result: dict, min_ratings: int, k: int, seed: int) -> bool:
    cfg = result["config"]
    return cfg["min_ratings"] == min_ratings and cfg["k"] == k and cfg["seed"] == seed


def show_stale_warning(min_ratings: int, k: int, seed: int) -> None:
    result = get_result()
    if result is None:
        return
    if not params_match_result(result, min_ratings, k, seed):
        cfg = result["config"]
        st.warning(
            f"Displayed results used min ratings {cfg['min_ratings']}, K={cfg['k']}, seed {cfg['seed']}. "
            "Sidebar values have changed. Run analysis again to refresh."
        )


def assignment_column_config() -> dict:
    return {
        "movieId": st.column_config.NumberColumn("Movie ID", format="%d"),
        "title": st.column_config.TextColumn("Title", width="large"),
        "cluster": st.column_config.NumberColumn("GMM cluster", format="%d"),
        "kmeans_cluster": st.column_config.NumberColumn("K-Means cluster", format="%d"),
        "mean_rating": st.column_config.NumberColumn("Mean rating", format="%.2f"),
        "std_rating": st.column_config.NumberColumn("Rating SD", format="%.2f"),
        "n_ratings": st.column_config.NumberColumn("Ratings", format="%d"),
        "max_responsibility": st.column_config.ProgressColumn(
            "Max responsibility", min_value=0, max_value=1, format="%.3f"
        ),
        "genres": st.column_config.TextColumn("Genres", width="medium"),
    }


def page_overview(data, overview, min_ratings, k, seed) -> None:
    st.caption("Machine learning practical mini-project · Roll 68")
    st.title("Segment movies by how audiences rate them")
    st.markdown(
        "This lab groups films from a local MovieLens-style catalog using a "
        "**Gaussian mixture model fitted by expectation-maximization**. "
        "Each movie is a 10-dimensional histogram: the share of its *observed* "
        "ratings at 0.5 through 5.0 stars. Missing user–movie pairs are left missing, not filled with zero. "
        "K-Means on the same matrix is the baseline, not a replacement for the mixture."
    )

    left, right = st.columns((1.35, 1), gap="large")
    with left:
        st.subheader("What the dataset contains")
        st.markdown(
            f"""
            | | |
            | --- | ---: |
            | Ratings | {overview['n_ratings']:,} |
            | Users | {overview['n_users']:,} |
            | Catalog titles | {overview['n_movies_catalog']:,} |
            | Titles with at least one rating | {overview['n_movies_rated']:,} |
            | Unrated catalog titles | {overview['n_movies_unrated']:,} |
            | Rating scale | {overview['rating_min']:g}–{overview['rating_max']:g} |
            """
        )
        st.caption(f"Files are read from `{RAW_DIR}`. Tags and links are not used as clustering features.")
    with right:
        st.subheader("Model configuration")
        result = get_result()
        if result is None:
            st.markdown(
                f"""
                **Ready to fit, not yet run.**

                - Minimum ratings per movie: **{min_ratings}**
                - Mixture components K: **{k}**
                - Random seed: **{seed}**
                - Covariance: diagonal
                - GMM initialisations: 3 (K-Means starts)
                """
            )
            st.caption("Use **Run analysis** in the sidebar. Changing pages does not refit the model.")
        else:
            cfg = result["config"]
            gmm = result["gmm"]
            st.markdown(
                f"""
                **Last fitted run**

                - Movies retained: **{result['prep_stats']['n_movies_retained']:,}**
                - Movies below the rating floor: **{result['prep_stats']['n_movies_excluded']:,}**
                - K = **{cfg['k']}**, min ratings = **{cfg['min_ratings']}**, seed **{cfg['seed']}**
                - GMM converged: **{fmt_bool(gmm.converged)}** in **{gmm.n_iter}** EM iterations
                - PCA (display only): {result['pca_var'][0]:.1%} / {result['pca_var'][1]:.1%} variance on axes 1 and 2
                """
            )

    st.subheader("How a run is built")
    steps = [
        ("Load", "Read local `ratings.csv` and `movies.csv`. Original files are never overwritten."),
        ("Filter", "Keep movies with at least the chosen number of observed ratings."),
        ("Features", "Turn each movie into rating-level proportions that sum to one."),
        ("Fit", "Estimate a diagonal Gaussian mixture with EM; fit K-Means on the same matrix."),
        ("Read", "Inspect sizes, profiles, AIC/BIC (GMM only), silhouette of hard labels, and exports."),
    ]
    cols = st.columns(5)
    for col, (title, body) in zip(cols, steps):
        with col:
            st.markdown(f"**{title}**")
            st.caption(body)

    st.subheader("How to read the result")
    result = get_result()
    if result is None:
        st.markdown(
            "After fitting, clusters describe **rating-pattern families**, not quality ranks or genres. "
            "Overlapping clouds are expected: a mixture can share movies across components through responsibilities. "
            "A modest or negative silhouette means hard assignments overlap in Euclidean space; it does not by itself "
            "mean the mixture is useless."
        )
        return

    gmm = result["gmm"]
    km = result["kmeans"]
    rec = result["recommendation"]
    m1, m2, m3 = st.columns(3)
    m1.metric("GMM silhouette (hard labels)", f"{gmm.silhouette:.3f}" if gmm.silhouette is not None else "—")
    m2.metric("K-Means silhouette", f"{km.silhouette:.3f}" if km.silhouette is not None else "—")
    m3.metric("BIC-preferred K", str(rec["k"]))
    st.markdown(
        f"{rec['reason']} The selected K in this session is **{result['config']['k']}**, which may differ from the BIC pick "
        "if you chose it deliberately. Silhouette, AIC, and BIC answer different questions; they are not interchangeable scores."
    )
    st.caption("Continue in Cluster results for profiles, Cluster distribution for the PCA view, and Model comparison for GMM versus K-Means.")


def page_explorer(data, overview, min_ratings: int) -> None:
    st.title("Dataset explorer")
    st.markdown(
        "All figures below are computed from the local CSVs. Nothing is downloaded or invented. "
        "Tags and links are shown for completeness; they do not enter the clustering features."
    )

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("ratings.csv rows", f"{len(data['ratings']):,}")
    q2.metric("movies.csv rows", f"{len(data['movies']):,}")
    q3.metric("Unique users", f"{overview['n_users']:,}")
    q4.metric("Unrated catalog titles", f"{overview['n_movies_unrated']:,}")

    if overview["n_ratings_unmatched"]:
        st.warning(f"{overview['n_ratings_unmatched']} ratings have no matching movie title.")
    else:
        st.success("Every rating row joins to a `movieId` in movies.csv.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ratings.csv")
        st.dataframe(
            profile_frame(data["ratings"]),
            hide_index=True,
            width="stretch",
            alt="Column profile for ratings.csv",
        )
        st.caption(f"Duplicate rows: {overview['duplicate_ratings']}")
    with c2:
        st.subheader("movies.csv")
        st.dataframe(
            profile_frame(data["movies"]),
            hide_index=True,
            width="stretch",
            alt="Column profile for movies.csv",
        )
        st.caption(f"Duplicate rows: {overview['duplicate_movies']}")

    st.subheader("Rating patterns")
    chart(rating_frequency(data["ratings"]), "Counts of each half-star rating in the catalog")
    counts = data["ratings"].groupby("movieId").size()
    means = data["ratings"].groupby("movieId")["rating"].mean()
    c3, c4 = st.columns(2)
    with c3:
        chart(ratings_per_movie(counts), "Histogram of ratings per movie")
    with c4:
        chart(mean_rating_distribution(means), "Histogram of mean movie ratings")
    st.caption(
        f"Ratings per movie range from {int(counts.min())} to {int(counts.max())} "
        f"(median {float(counts.median()):.0f}). "
        f"{overview['n_movies_unrated']} catalog titles have no ratings at all."
    )

    st.subheader("Preprocessing at the current rating floor")
    st.markdown(
        f"Movies need at least **{min_ratings}** observed ratings to enter the feature matrix. "
        "That threshold is applied before clustering. Unobserved ratings are omitted, not imputed."
    )
    feats, stats, excluded = cached_feature_preview(min_ratings)
    p1, p2, p3 = st.columns(3)
    p1.metric("Movies retained", f"{stats['n_movies_retained']:,}")
    p2.metric("Excluded (too few ratings)", f"{stats['n_movies_excluded']:,}")
    p3.metric("Invalid or dropped rating rows", f"{stats['n_invalid_or_dropped_ratings']:,}")
    st.caption(
        f"Feature row sums range from {stats['feature_row_sum_min']:.6f} to {stats['feature_row_sum_max']:.6f} "
        f"across {stats['n_feature_dims']} share dimensions. Unmatched titles: {stats['unmatched_titles']}."
    )
    with st.expander("Preview movies below the rating floor"):
        st.caption("These titles are rated, but not often enough to be clustered at the current threshold.")
        st.dataframe(
            excluded.head(250),
            hide_index=True,
            width="stretch",
            height=320,
            alt="Movies excluded because they have too few ratings",
            column_config=assignment_column_config(),
        )

    st.subheader("Inspect a movie’s observed ratings")
    q = st.text_input("Search title", placeholder="e.g. Toy Story", key="explorer_search")
    movies = data["movies"]
    if q.strip():
        hits = movies[movies["title"].str.contains(q, case=False, na=False)].head(25)
        st.dataframe(hits, hide_index=True, width="stretch", alt="Title search matches", height=240)
        if len(hits):
            chosen = st.selectbox("Select movie", hits["title"].tolist())
            mid = int(hits.loc[hits["title"] == chosen, "movieId"].iloc[0])
            rs = data["ratings"][data["ratings"]["movieId"] == mid]
            if rs.empty:
                st.warning("This catalog title has no ratings in ratings.csv.")
            else:
                st.caption(
                    f"{len(rs)} observed ratings; mean {rs['rating'].mean():.3f}. "
                    "Users who never rated this title are not treated as zeros."
                )
                chart(
                    movie_rating_histogram(rs, f"Observed ratings for {chosen}"),
                    f"Star-rating counts for {chosen}",
                )
    else:
        st.caption("Type part of a title to preview its raw rating histogram.")

    with st.expander("Sample rows"):
        t1, t2 = st.tabs(["Ratings", "Movies"])
        with t1:
            st.dataframe(data["ratings"].head(40), hide_index=True, width="stretch", alt="First rows of ratings.csv")
        with t2:
            st.dataframe(data["movies"].head(40), hide_index=True, width="stretch", alt="First rows of movies.csv")


def page_lab(min_ratings: int, k: int, seed: int) -> None:
    st.title("EM clustering lab")
    st.markdown(
        "The movie analysis uses **`sklearn.mixture.GaussianMixture`**, which runs EM internally. "
        "A separate one-dimensional from-scratch demo lives on the Methodology page and is **not** "
        "the algorithm that assigns MovieLens titles."
    )

    st.subheader("What the sidebar controls change")
    a, b, c = st.columns(3)
    with a:
        st.markdown("**Minimum ratings**")
        st.caption(
            f"Currently **{min_ratings}**. Movies with fewer observed ratings are dropped before fitting. "
            "Raising it stabilises histograms and shrinks the set; lowering it includes noisier profiles."
        )
    with b:
        st.markdown("**Number of clusters, K**")
        st.caption(
            f"Currently **{k}**. This is the number of Gaussian components. "
            "The lab also fits K = 2…8 for AIC/BIC comparison, but displayed assignments use this K."
        )
    with c:
        st.markdown("**Random seed**")
        st.caption(
            f"Currently **{seed}**. It controls GMM and K-Means initialisation and the silhouette subsample. "
            "Keep it fixed for viva reproducibility."
        )
    st.caption(
        "Fixed in code, not exposed as widgets: diagonal covariance, `n_init=3` K-Means starts, "
        "`max_iter=200`, `reg_covar=1e-6`. Ordinary widget movement does not refit; use **Run analysis**."
    )

    result = require_result()
    if result is None:
        return

    ps = result["prep_stats"]
    gmm = result["gmm"]
    cfg = result["config"]

    st.subheader("Fit diagnostics")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Movies retained", f"{ps['n_movies_retained']:,}")
    d2.metric("Average log-likelihood", f"{gmm.inertia_or_ll:.4f}")
    d3.metric("Converged", fmt_bool(gmm.converged))
    d4.metric("EM iterations", str(gmm.n_iter))
    e1, e2, e3 = st.columns(3)
    e1.metric("AIC", f"{gmm.aic:,.1f}")
    e2.metric("BIC", f"{gmm.bic:,.1f}")
    e3.metric("Covariance", cfg["covariance_type"])
    st.caption(
        "Log-likelihood, AIC, and BIC belong to the Gaussian mixture. They are not defined for the K-Means baseline. "
        "Convergence means the EM lower bound stopped changing within sklearn’s tolerance before `max_iter`."
    )

    st.subheader("Component means")
    st.markdown(
        "Each row is a fitted Gaussian mean in rating-share space. These are model parameters, "
        "not the same as the empirical cluster averages (those appear on Cluster results)."
    )
    chart(gmm_means_heatmap(gmm.model.means_), "Heatmap of fitted GMM component means")

    with st.expander("Feature matrix notes"):
        st.markdown(
            f"- Feature row-sum range: `{ps['feature_row_sum_min']:.6f}` – `{ps['feature_row_sum_max']:.6f}` "
            "(compositional shares, not scaled to unit variance).\n"
            f"- Unmatched titles after join: {ps['unmatched_titles']}.\n"
            f"- Invalid or dropped rating rows: {ps['n_invalid_or_dropped_ratings']}.\n"
            "- No additional feature scaling is applied beyond building proportions."
        )


def page_results() -> None:
    st.title("Cluster results")
    result = require_result()
    if result is None:
        return

    assigned = result["assigned"]
    summary = result["summary"]
    briefs = cluster_briefs(summary)

    st.markdown(
        "Hard labels are the MAP component (largest responsibility). "
        "A movie can still have mass on other components. "
        "The short readings below are taken from each cluster’s **mean rating-share profile** — not from genres."
    )

    st.subheader("Membership")
    chart(cluster_size_bar(summary), "Bar chart of movies per GMM cluster")

    st.subheader("What each cluster looks like")
    for row in briefs.itertuples():
        st.markdown(f"**{cluster_label(row.cluster)}** — {row.reading}")

    st.subheader("Rating-pattern profiles")
    chart(cluster_profiles(assigned), "Line chart of mean rating shares by cluster")
    profile = rating_profile_table(assigned)
    st.dataframe(
        profile,
        hide_index=True,
        width="stretch",
        alt="Numeric mean rating-share profile by cluster",
        column_config={
            col: st.column_config.NumberColumn(col, format="%.3f")
            for col in profile.columns
            if col != "cluster"
        },
    )

    show = summary.copy()
    show["share"] = show["share"].map(fmt_pct)
    st.subheader("Numeric summary")
    st.dataframe(
        show,
        hide_index=True,
        width="stretch",
        alt="Cluster summary statistics",
        column_config={
            "mean_rating": st.column_config.NumberColumn("Mean rating", format="%.3f"),
            "mean_std_rating": st.column_config.NumberColumn("Mean SD", format="%.3f"),
            "mean_n_ratings": st.column_config.NumberColumn("Mean ratings", format="%.1f"),
            "mean_max_responsibility": st.column_config.NumberColumn("Mean max responsibility", format="%.3f"),
        },
    )

    st.subheader("Representative movies")
    st.caption("Highest MAP responsibility within each cluster, breaking ties toward more frequently rated titles.")
    st.dataframe(
        result["representatives"],
        hide_index=True,
        width="stretch",
        alt="Representative movies for each cluster",
        column_config=assignment_column_config(),
        height=360,
    )

    st.subheader("Search assignments")
    q = st.text_input("Filter by title", key="cluster_search", placeholder="e.g. Matrix")
    view = assigned[["movieId", "title", "cluster", "mean_rating", "n_ratings", "max_responsibility", "genres"]]
    if q.strip():
        view = view[view["title"].str.contains(q, case=False, na=False)]
    clusters = sorted(assigned["cluster"].unique().tolist())
    cl_opt = st.multiselect("Clusters", clusters, format_func=cluster_label)
    if cl_opt:
        view = view[view["cluster"].isin(cl_opt)]
    st.caption(f"{len(view):,} movies in the current filter.")
    st.dataframe(
        view.sort_values(["cluster", "max_responsibility"], ascending=[True, False]),
        hide_index=True,
        width="stretch",
        height=420,
        alt="Searchable movie cluster assignments",
        column_config=assignment_column_config(),
    )


def page_distribution() -> None:
    st.title("Cluster distribution")
    result = require_result()
    if result is None:
        return

    assigned = result["assigned"]
    X = result["X"]
    gmm = result["gmm"]
    labels = gmm.labels
    var1, var2 = result["pca_var"]

    st.subheader("Two-dimensional PCA view")
    st.markdown(
        f"Clustering is performed in the original **{X.shape[1]}-D rating-share space**. "
        f"The scatter below is a PCA projection used only for display. "
        f"The two axes capture **{var1:.1%}** and **{var2:.1%}** of feature variance "
        f"({var1 + var2:.1%} together). Overlap here does not prove overlap in 10-D, and separation here "
        "does not prove separation in 10-D. Ellipses are empirical 95% regions of the **projected** points, "
        "not the GMM covariances in the original space."
    )
    chart(
        pca_scatter_with_ellipses(assigned),
        "PCA scatter of movies with two-dimensional cluster spread ellipses",
    )
    with st.expander("Projection without ellipses"):
        chart(pca_scatter(assigned), "PCA scatter of movies coloured by GMM cluster")

    st.subheader("Population")
    chart(cluster_size_bar(result["summary"]), "Cluster population bar chart")

    st.subheader("Spread in the original feature space")
    means = gmm.model.means_
    dists = np.linalg.norm(X - means[labels], axis=1)
    chart(
        distance_to_mean_box(assigned, dists),
        "Box plots of Euclidean distance to the assigned GMM mean",
    )
    ddf = assigned[["cluster"]].copy()
    ddf["distance_to_component_mean"] = dists
    table = (
        ddf.groupby("cluster")["distance_to_component_mean"]
        .agg(median="median", iqr=lambda s: s.quantile(0.75) - s.quantile(0.25), mean="mean")
        .reset_index()
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        alt="Median and IQR of distances to GMM component means",
        column_config={
            "median": st.column_config.NumberColumn("Median distance", format="%.4f"),
            "iqr": st.column_config.NumberColumn("IQR", format="%.4f"),
            "mean": st.column_config.NumberColumn("Mean distance", format="%.4f"),
        },
    )
    st.caption("Distances are Euclidean in the ten rating-share dimensions, not probabilities.")

    st.subheader("Silhouette of hard labels")
    try:
        ss = silhouette_samples(X, labels, metric="euclidean")
        overall = silhouette_score(X, labels, metric="euclidean")
        st.metric("Overall silhouette", f"{overall:.4f}")
        chart(silhouette_by_cluster_box(labels, ss), "Per-cluster silhouette distributions")
    except Exception as exc:
        st.warning(str(exc))
    st.caption(
        "Positive values: closer to own cluster than to the nearest other cluster. "
        "Near zero: on a boundary. Negative: closer to another cluster. "
        "Silhouette is not a GMM probability."
    )

    st.subheader("Assignment certainty")
    chart(max_responsibility_box(assigned), "Box plots of maximum GMM responsibility by cluster")
    st.caption("Higher maximum responsibility means a sharper MAP assignment, not that the cluster is “correct”.")

    st.subheader("Rating-profile comparison")
    chart(rating_profile_heatmap(assigned), "Heatmap of mean rating shares by cluster")
    st.caption("Cells are empirical mean shares of observed ratings, not GMM densities.")


def page_compare() -> None:
    st.title("Model comparison")
    result = require_result()
    if result is None:
        return

    cmp_df = result["comparison"]
    rec = result["recommendation"]
    st.markdown(
        "GMM models for K = 2 through 8 are fitted on the **same** rating-share matrix as K-Means. "
        "**AIC and BIC apply only to the mixture** (they use the Gaussian likelihood and a parameter count). "
        "K-Means has no AIC/BIC here. **Silhouette** scores a hard partition in Euclidean space for both methods. "
        "K-Means **inertia** is within-cluster sum of squares; GMM reports average log-likelihood instead. "
        "Do not rank the two algorithms on a metric that only one of them defines."
    )

    display = cmp_df.copy()
    display["gmm_sizes"] = display["gmm_sizes"].astype(str)
    display["kmeans_sizes"] = display["kmeans_sizes"].astype(str)
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        alt="GMM and K-Means metrics for K from 2 to 8",
        column_config={
            "gmm_log_likelihood": st.column_config.NumberColumn("GMM mean log-likelihood", format="%.4f"),
            "gmm_aic": st.column_config.NumberColumn("GMM AIC", format="%.1f"),
            "gmm_bic": st.column_config.NumberColumn("GMM BIC", format="%.1f"),
            "gmm_silhouette": st.column_config.NumberColumn("GMM silhouette", format="%.3f"),
            "kmeans_inertia": st.column_config.NumberColumn("K-Means inertia", format="%.3f"),
            "kmeans_silhouette": st.column_config.NumberColumn("K-Means silhouette", format="%.3f"),
        },
    )

    chart(aic_bic_plot(cmp_df), "AIC and BIC curves for GMM over K")
    chart(silhouette_compare(cmp_df), "Grouped bar chart of GMM and K-Means silhouette versus K")
    st.info(rec["reason"])
    st.caption(
        f"Silhouette uses the full matrix when n ≤ {result['config']['silhouette_max_samples']}, "
        "otherwise a fixed-seed sample of that size."
    )

    k = result["config"]["k"]
    st.subheader(f"Head-to-head at selected K = {k}")
    gmm, km = result["gmm"], result["kmeans"]
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("GMM silhouette", f"{gmm.silhouette:.4f}" if gmm.silhouette is not None else "—")
    h2.metric("K-Means silhouette", f"{km.silhouette:.4f}" if km.silhouette is not None else "—")
    h3.metric("K-Means inertia", f"{km.inertia_or_ll:,.3f}")
    h4.metric("GMM mean log-likelihood", f"{gmm.inertia_or_ll:.4f}")
    st.markdown(
        "A higher K-Means silhouette at this K means the **hard Voronoi partition** is more compact/separated. "
        "It does not imply K-Means is a better generative model of rating histograms. "
        "GMM remains the assigned EM method because it yields soft membership and a proper likelihood."
    )
    chart(gmm_vs_kmeans_sizes(gmm.sizes, km.sizes), "Grouped bar chart of GMM versus K-Means cluster sizes")

    st.subheader("Worked silhouette for one movie")
    m = result["manual_silhouette"]
    match = "matches" if m["matches_sklearn"] else "does not match"
    st.markdown(
        f"Movie **{m['title']}** (`movieId` {m['movieId']}) is in cluster {m['cluster']}. "
        f"a(i) = {m['a']:.6f} (mean distance to others in its cluster). "
        f"b(i) = {m['b']:.6f} (mean distance to the nearest other cluster, {cluster_label(m['nearest_other_cluster'])}). "
        f"s(i) = (b − a) / max(a, b) = **{m['s']:.6f}**. "
        f"sklearn `silhouette_samples` = {m['sklearn_s']:.6f} ({match})."
    )


def page_method(seed: int) -> None:
    st.title("Methodology")
    st.markdown(
        """
The production pipeline is:

**Local CSVs → cleaning and the rating-count floor → rating-distribution features → EM/GMM clustering → evaluation → interpretation.**

K-Means is fitted on the same feature matrix as a baseline. PCA is computed only to draw a scatter plot.
        """
    )

    st.subheader("Pipeline in this repository")
    st.markdown(
        """
1. **Dataset.** `data/raw/ratings.csv` and `movies.csv` are loaded as-is. `tags.csv` and `links.csv` are inspected, not modelled.
2. **Cleaning.** Drop rows with missing `movieId` or `rating`, and keep only the ten legal half-star values. Duplicate checks are reported; originals are never overwritten.
3. **Filtering.** A movie enters the model only if it has at least *min ratings* observed ratings (default 20).
4. **Features.** For each retained movie, counts at 0.5…5.0 are divided by that movie’s own rating count. Unobserved pairs are **not** imputed. No extra standardisation is applied; the vector already lives on a simplex.
5. **EM / GMM.** `sklearn.mixture.GaussianMixture` with diagonal covariance, three K-Means initialisations, and a fixed seed. The E-step computes responsibilities; the M-step updates weights, means, and variances until the lower bound converges.
6. **Baseline.** `sklearn.cluster.KMeans` with the same K and seed.
7. **Evaluation.** AIC and BIC for GMM over K = 2…8; silhouette on hard labels for both algorithms; a manual silhouette identity for one movie.
8. **Interpretation.** Clusters are described from empirical rating-share profiles and mean ratings, not from assumed genre names.
        """
    )

    st.subheader("Model, in accessible language")
    st.markdown(
        """
A Gaussian mixture says each movie’s histogram is more likely under some of K ellipsoidal blobs than others.
EM is the fitting procedure: guess membership, update blob parameters, repeat.
**Diagonal covariance** means each rating bin has its own variance inside a component, with no modelled
correlation between bins — a stability choice on 10-D shares. `reg_covar` stops a component from collapsing to a spike.

**AIC** = 2p − 2 ln L. **BIC** = p ln n − 2 ln L. Lower is better; BIC penalises extra components more.
Neither is computed for K-Means in this project.
        """
    )

    st.subheader("From-scratch EM (synthetic, 1-D)")
    st.warning(
        "This demonstration is educational. It is **not** the model that produced the movie clusters. "
        "Movie clustering uses scikit-learn’s multidimensional GaussianMixture."
    )
    demo = run_demo(seed=seed)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Iterations", str(demo.n_iter))
    m2.metric("Converged", fmt_bool(demo.converged))
    m3.metric("Final mean log-likelihood", f"{demo.log_likelihoods[-1]:.4f}")
    m4.metric("Means", ", ".join(f"{m:.3f}" for m in demo.means))
    chart(em_ll_curve(demo.log_likelihoods), "Log-likelihood curve for the synthetic one-dimensional EM demo")
    st.code(
        "E-step:  γ_ik ∝ π_k N(x_i | μ_k, σ_k²)\n"
        "M-step:  π_k = mean(γ_ik);  μ_k = Σ_i γ_ik x_i / Σ_i γ_ik;  σ_k² similarly, floored at 1e-4",
        language="text",
    )
    st.caption("Synthetic samples are drawn around −2.0 and +2.2. That geometry is unrelated to MovieLens.")


def page_export() -> None:
    st.title("Report and exports")
    st.markdown(
        "Academic write-ups stay in `report/`. Analysis CSVs and charts are **results of a fit**, "
        "not replacements for the original MovieLens files."
    )

    report_files = [
        (ROOT / "report" / "mini_project_report.md", "Project report (Markdown)", "mini_project_report.md"),
        (ROOT / "report" / "viva_questions.md", "Viva questions", "viva_questions.md"),
        (ROOT / "report" / "rubric_audit.md", "Rubric audit", "rubric_audit.md"),
    ]
    st.subheader("Academic documents")
    for path, label, filename in report_files:
        if path.exists():
            st.download_button(
                label,
                path.read_text(encoding="utf-8"),
                file_name=filename,
                mime="text/markdown",
                key=f"dl_{filename}",
                width="stretch",
            )
        else:
            st.caption(f"Missing: {path.name}")

    result = get_result()
    st.subheader("Fitted-result downloads")
    if result is None:
        st.info("Run analysis to enable CSV exports of cluster assignments and comparison tables.")
    else:
        assigned = result["assigned"]
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
        ] + [c for c in assigned.columns if str(c).startswith("resp_c")]
        st.download_button(
            "Movie cluster assignments (CSV)",
            assigned[export_cols].to_csv(index=False).encode("utf-8"),
            file_name="movie_cluster_assignments.csv",
            mime="text/csv",
            key="dl_assign",
            width="stretch",
        )
        st.download_button(
            "Model comparison (CSV)",
            result["comparison"].to_csv(index=False).encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv",
            key="dl_cmp",
            width="stretch",
        )
        st.download_button(
            "Cluster summary (CSV)",
            result["summary"].to_csv(index=False).encode("utf-8"),
            file_name="cluster_summary.csv",
            mime="text/csv",
            key="dl_sum",
            width="stretch",
        )
        st.caption(
            "Assignments include hard GMM cluster, K-Means cluster, and component responsibilities. "
            "Original `ratings.csv` / `movies.csv` are never written."
        )
        if st.button("Write files into output/", type="primary"):
            try:
                paths = persist_outputs(result)
                st.success("Wrote analysis CSVs and chart HTML under output/.")
                st.dataframe(
                    pd.DataFrame({"artifact": list(paths.keys()), "path": list(paths.values())}),
                    hide_index=True,
                    width="stretch",
                    alt="Paths of files written under output/",
                )
            except Exception as exc:
                st.error(f"Could not write outputs: {exc}")

    st.subheader("Files already on disk")
    existing = []
    for folder in (ROOT / "output" / "results", ROOT / "output" / "tables", ROOT / "output" / "figures"):
        if folder.exists():
            for path in sorted(folder.glob("*")):
                if path.is_file():
                    existing.append({"folder": folder.name, "file": path.name, "bytes": path.stat().st_size})
    if existing:
        st.dataframe(
            pd.DataFrame(existing),
            hide_index=True,
            width="stretch",
            alt="Existing output artifacts on disk",
        )
    else:
        st.caption("No files under output/ yet.")


def main() -> None:
    inject_brand()
    missing = missing_raw_files()

    with st.sidebar:
        st.markdown("**Cinema intelligence**")
        st.caption("Movie Clustering Analytics Lab · EM on rating distributions · Roll 68")
        if missing:
            st.error("Dataset files missing: " + ", ".join(missing))
        else:
            data_preview = cached_raw()
            overview_preview = dataset_overview(data_preview)
            st.caption(f"{overview_preview['n_ratings']:,} ratings · {overview_preview['n_users']} users")

        page = st.radio(
            "Page",
            options=[key for key, _ in NAV],
            format_func=lambda key: NAV_LABELS[key],
        )

        st.subheader("Analysis")
        min_ratings = st.slider(
            "Minimum ratings per movie",
            min_value=5,
            max_value=50,
            value=DEFAULT_MIN_RATINGS,
            step=1,
            help="Movies with fewer observed ratings are excluded before clustering.",
        )
        k = st.slider(
            "Number of clusters, K",
            min_value=2,
            max_value=8,
            value=DEFAULT_K,
            step=1,
            help="Number of Gaussian components. Assignments use this K; AIC/BIC still scan 2 through 8.",
        )
        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=10_000,
            value=DEFAULT_SEED,
            step=1,
            help="Fixes GMM/K-Means initialisation and silhouette sampling.",
        )
        st.session_state["seed_ctrl"] = int(seed)
        run = st.button("Run analysis", type="primary", width="stretch")
        result = get_result()
        if result is not None:
            if params_match_result(result, int(min_ratings), int(k), int(seed)):
                st.success("Results match the current controls.")
            else:
                st.warning("Controls differ from the last run.")

    if missing:
        st.error(f"Place the MovieLens CSVs in `{RAW_DIR}`.")
        return

    data = cached_raw()
    overview = dataset_overview(data)

    if run:
        with st.spinner("Fitting GMM (K=2…8) and K-Means on rating-share features…"):
            try:
                st.session_state["result"] = run_analysis(
                    min_ratings=int(min_ratings), k=int(k), seed=int(seed), data=data
                )
                st.toast("Analysis complete.", icon=":material/check_circle:")
            except Exception as exc:
                st.exception(exc)
                return

    show_stale_warning(int(min_ratings), int(k), int(seed))

    if page == "overview":
        page_overview(data, overview, int(min_ratings), int(k), int(seed))
    elif page == "explorer":
        page_explorer(data, overview, int(min_ratings))
    elif page == "lab":
        page_lab(int(min_ratings), int(k), int(seed))
    elif page == "results":
        page_results()
    elif page == "distribution":
        page_distribution()
    elif page == "compare":
        page_compare()
    elif page == "method":
        page_method(int(seed))
    else:
        page_export()


main()
