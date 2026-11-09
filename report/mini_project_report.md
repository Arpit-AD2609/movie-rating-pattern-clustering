# Movie Segmentation Based on User Ratings Using Expectation-Maximization Clustering

**Programme:** Machine Learning Practical Mini-Project (A.Y. 2025–26)  
**Roll number:** 68  
**Assigned problem:** 68 — Segment movies by user ratings dataset using EM clustering  
**Application title:** MOVIE CLUSTERING LAB  
**Date:** 28 October 2026  

---

## Abstract

This project groups movies by *how* users rate them, not by a single average score. Each movie with at least 20 observed ratings is represented by the empirical distribution of those ratings on the half-star scale 0.5–5.0. A Gaussian mixture model, whose parameters are estimated by Expectation-Maximization, is fitted in that 10-dimensional simplex. K from 2 to 8 is compared using AIC and BIC; both criteria select **K = 7**. K-Means on the same features is the baseline. A separate 1-D EM implementation on synthetic data is included for the algorithm-from-scratch requirement and is not used as the MovieLens model.

## 1. Introduction

Collaborative-filtering data are sparse: most users never rate most movies. Filling those gaps with zero would invent “ratings” that never occurred. The analysis therefore uses only observed ratings, aggregated into a histogram per movie. Movies with similar histograms—similar mixes of 1-star and 5-star reactions—are placed in the same mixture component. The deliverable is a local Streamlit lab plus this report, a notebook, and exported tables.

## 2. Objectives

1. Describe the supplied MovieLens tables (counts, types, missingness, joins).
2. Build valid rating-share features and document the minimum-rating filter.
3. Fit GMM/EM, tune K, and record convergence.
4. Implement EM from scratch on synthetic 1-D data.
5. Compare with K-Means and compute silhouette, including one fully manual example.
6. Present interactive charts and a written interpretation tied to observed profiles.

## 3. Dataset

Files (unchanged): `data/raw/ratings.csv`, `movies.csv`, `tags.csv`, `links.csv`.

| Property | Value |
| --- | --- |
| Rating rows | 100,836 |
| Unique users | 610 |
| Movie catalog rows | 9,742 |
| Movies with at least one rating | 9,724 |
| Catalog movies with zero ratings | 18 |
| Ratings with no matching title | 0 |
| Duplicate rating/movie rows | 0 / 0 |
| Missing values in ratings or movie columns | none |
| Rating values | 0.5, 1.0, …, 5.0 |
| Tags / links (not used in the model) | 3,683 / 9,742 |

There is **no supervised target**. Clustering is unsupervised. Tags and TMDb/IMDb links are available for inspection only.

Ratings-per-movie are heavy-tailed (median 3, maximum 329). A threshold of 20 ratings leaves **1,297** movies and excludes **8,427** rarely rated titles so that histograms are not dominated by a handful of votes.

## 4. Preprocessing and features

1. Drop rows with missing `movieId` or `rating`; keep ratings in the official half-star set (none were invalid in this file).
2. Count ratings at each of the 10 levels per `movieId`.
3. Convert counts to proportions. Row sums are in \([1-10^{-15},\,1+10^{-15}]\).
4. Left-join titles and genres on `movieId` (0 unmatched among retained movies).
5. **Do not** impute unobserved user–movie pairs. **Do not** add genre, tag, or popularity covariates to the GMM feature matrix.

Auxiliary columns `n_ratings`, `mean_rating`, and `std_rating` are stored for tables and plots only.

No scaling is applied: features are already shares on a common simplex. Outliers are not removed; extreme histograms are legitimate rating patterns.

## 5. Methodology (GMM and EM)

A mixture of \(K\) Gaussians is

\[
p(x) = \sum_{k=1}^{K} \pi_k\,\mathcal{N}(x\mid\mu_k,\Sigma_k).
\]

EM maximises the observed-data likelihood:

- **E-step:** \(\gamma_{ik} = p(z_i=k\mid x_i)\).
- **M-step:** \(\pi_k,\mu_k,\Sigma_k\) are weighted means and covariances of the \(x_i\).

**MovieLens model:** `sklearn.mixture.GaussianMixture`, `covariance_type='diag'`, `n_init=3`, `init_params='kmeans'`, `random_state=42`, `max_iter=200`, `reg_covar=1e-6`. Diagonal covariance is a modelling choice: 10-D compositional vectors with sparse histograms make unconstrained full covariance ill-conditioned.

**From-scratch EM:** two 1-D Gaussians on 400 synthetic points (\(N(-2,0.6^2)\) and \(N(2.2,0.7^2)\)). Variances are floored at \(10^{-4}\). Responsibilities are computed in the log domain. This demo recovered means \(-2.022\) and \(2.209\) in 7 iterations (seed 42) and is **not** the movie model.

**PCA** (2 components, 44.7% + 17.7% variance) is visualisation only.

**Information criteria:** \(\mathrm{AIC}=2p-2\ln L\), \(\mathrm{BIC}=p\ln n-2\ln L\). Lower is better.

**Silhouette** uses hard MAP labels and Euclidean distance. If \(n>1500\), a seed-42 subsample of 1500 would be used; here \(n=1297\), so the full matrix is scored.

## 6. Model configuration and tuning

| Setting | Value |
| --- | --- |
| Minimum ratings | 20 |
| Feature dimension | 10 |
| Movies modelled | 1,297 |
| K grid | 2, 3, 4, 5, 6, 7, 8 |
| Selected K | **7** (min AIC and min BIC) |
| Baseline | K-Means, `n_init=10`, same K and seed |

GMM comparison (seed 42):

| K | Mean log-lik. | AIC | BIC | Converged | EM iters | GMM silhouette | K-Means silhouette |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 16.403 | −42469 | −42257 | yes | 7 | 0.248 | 0.282 |
| 3 | 17.474 | −45204 | −44883 | yes | 28 | 0.042 | 0.211 |
| 4 | 18.393 | −47547 | −47118 | yes | 20 | 0.025 | 0.203 |
| 5 | 18.315 | −47302 | −46765 | yes | 34 | −0.020 | 0.191 |
| 6 | 18.972 | −48966 | −48320 | yes | 43 | −0.048 | 0.189 |
| **7** | **19.335** | **−49865** | **−49111** | **yes** | **32** | **−0.038** | **0.180** |
| 8 | 19.154 | −49353 | −48490 | yes | 26 | −0.057 | 0.171 |

K = 5 and K = 8 have slightly lower likelihood than a neighbour despite three initialisations; GMM remains locally optimal. BIC still prefers 7 over 8.

Silhouette is highest at K = 2. That does **not** contradict BIC: silhouette rewards separated hard blobs, while BIC scores the mixture density. Rating-share vectors overlap, so a useful GMM can have a near-zero or negative silhouette.

## 7. Results (K = 7)

- GMM average log-likelihood: 19.335  
- AIC −49865; BIC −49111  
- Converged: yes, 32 iterations  
- GMM silhouette: −0.038  
- K-Means inertia: 26.307; K-Means silhouette: 0.180  
- GMM sizes: 204, 139, 195, 96, 203, 336, 124  
- Mean max-responsibility per cluster is 0.93–0.97, so MAP labels are typically confident even though clusters overlap in PCA.

### Manual silhouette (real movie, not illustrative)

Movie **Bullets Over Broadway (1994)** (`movieId` 348), GMM cluster 4:

- \(a(i)=0.262539\) — mean Euclidean distance to other movies in cluster 4  
- \(b(i)=0.260490\) — mean distance to cluster 3 (nearest other)  
- \(s(i)=(b-a)/\max(a,b)=-0.007805\)  
- `sklearn.metrics.silhouette_samples` = −0.007805 (matches)

A negative \(s(i)\) means this title sits slightly closer to another component’s members than to its own MAP cluster—consistent with overlapping mixtures.

### Cluster profiles (mean share of stars)

Clusters are named only by observed statistics, not by genre.

| Cluster | n | Mean rating | Mean std | Mean #ratings | Distinctive rating shares |
| --- | --- | --- | --- | --- | --- |
| 0 | 204 | 3.46 | 0.90 | 45 | Almost no 0.5★; mass on 3–4★ |
| 1 | 139 | 2.69 | 1.02 | 40 | Highest 1★ and 2★ shares (0.085, 0.182) |
| 2 | 195 | 3.32 | 1.02 | 55 | Broad mix including low-star tail |
| 3 | 96 | 3.96 | 0.89 | 103 | Highest 5★ share (0.231); most frequently rated |
| 4 | 203 | 3.33 | 0.89 | 38 | Peak at 3★ (0.303); no 0.5★ or 1.5★ on average |
| 5 | 336 | 3.89 | 0.80 | 46 | Largest group; high 4★ share (0.330); lowest spread |
| 6 | 124 | 3.85 | 0.93 | 77 | Similar high-star shape to C5, with a small 0.5★ tail |

High-responsibility examples (not a quality ranking): C1 includes *Batman & Robin (1997)* and *Godzilla (1998)*; C3 includes *Casablanca (1942)*, *Memento (2000)*, *Inception (2010)*; C5 includes *Cinema Paradiso (1989)* and *My Neighbor Totoro (1988)*. Titles illustrate the **rating mix**, not a claim that the cluster *is* “classics” or “flops.”

Figures: `output/figures/rating_frequency.png`, `ratings_per_movie.png`, `mean_rating_distribution.png`, `aic_bic.png`, `cluster_sizes.png`, `pca_scatter.png`, `cluster_profiles.png`, `silhouette_compare.png`, `gmm_vs_kmeans_sizes.png`, `em_ll_curve.png`.

## 8. Comparison and analysis

K-Means partitions Euclidean space into Voronoi cells and therefore reports a higher silhouette (0.18 vs −0.04 at K = 7) and more even sizes. GMM models overlapping ellipsoids (here axis-aligned) and reports a higher likelihood / better AIC–BIC. For **segmentation of rating histograms**, the mixture is the assigned method: a movie can sit between “often 3-star” and “often 5-star” with a soft responsibility. K-Means is a useful baseline for compactness, not a replacement for EM on this brief.

## 9. Interpretation and relevance

Platforms can use rating-shape segments to:

- separate titles that are widely liked (mass on 4–5★) from titles with concentrated low scores;
- avoid treating a 3.3 average with a wide 1–5 spread as equivalent to a tight 3.3 average;
- communicate uncertainty via responsibilities rather than a single bucket.

This is not a recommender: no user-level prediction is produced.

## 10. Limitations

- MovieLens small (610 users) is not the full MovieLens catalogue.
- The 20-rating filter drops most titles; cold-start movies are out of scope.
- Diagonal Gaussians ignore feature correlations among adjacent star levels.
- Compositional data live on a simplex; a Dirichlet mixture could be more natural.
- AIC/BIC and silhouette disagree; K = 7 is a BIC choice, not a unique truth.
- Genre is unused in the model, so clusters are not genre taxonomies.

## 11. Conclusion

A working local lab fits GMM/EM to 1,297 movies encoded as observed rating shares. **K = 7** minimises AIC and BIC. K-Means is more compact by silhouette but does not model soft membership. A tested 1-D EM demo recovered known synthetic means. All metrics and figures come from the local CSVs; original files were not modified.

## References

1. Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *JRSS B*.
2. Reynolds, D. (2009). Gaussian mixture models. *Encyclopedia of Biometrics*.
3. Pedregosa et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*.
4. Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets. *ACM TiiS*.
5. Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation of cluster analysis.
6. Akaike, H. (1974). A new look at the statistical model identification.
7. Schwarz, G. (1978). Estimating the dimension of a model.
