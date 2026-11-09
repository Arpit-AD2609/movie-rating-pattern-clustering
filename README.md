# MOVIE CLUSTERING LAB

Movie Segmentation Using Expectation-Maximization  
Machine Learning Practical Mini-Project · Roll **68** · Problem **68**

Segment movies from the local MovieLens ratings files using a Gaussian mixture model (EM) on **rating-distribution features**. K-Means on the same matrix is the baseline.

## Run the app (Windows)

```powershell
cd "C:\ML MiniProject New"
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard opens at **http://localhost:8501**.

Use the sidebar: set minimum ratings (default 20), GMM components K (default **7**, BIC-selected), seed (42), then **Run Analysis**. Changing tabs does not refit the model.

## Notebook

```powershell
cd "C:\ML MiniProject New"
python -m jupyter notebook notebooks\movie_em_clustering.ipynb
```

Or execute all cells:

```powershell
python -m jupyter nbconvert --to notebook --execute notebooks\movie_em_clustering.ipynb --output movie_em_clustering.ipynb
```

## Project layout

- `app.py` — Streamlit dashboard
- `src/` — shared loader, features, GMM/K-Means, 1-D EM demo, charts
- `data/raw/` — original CSVs (do not overwrite)
- `notebooks/movie_em_clustering.ipynb`
- `report/` — markdown report, viva Q&A, rubric audit
- `output/figures|tables|results/` — charts and CSV exports of **results** only

## Data (computed locally)

| Item | Value |
| --- | --- |
| Ratings | 100,836 |
| Users | 610 |
| Catalog movies | 9,742 |
| Movies with ≥1 rating | 9,724 |
| Unrated catalog titles | 18 |
| Rating scale | 0.5–5.0 (half stars) |
| Movies with ≥20 ratings (analysis set) | 1,297 |

Features: 10 proportions \(p_{0.5},\ldots,p_{5.0}\) per movie. Unobserved user–movie pairs are **not** filled with zero.

## Selected model (seed 42, min ratings 20)

- Algorithm: `sklearn.mixture.GaussianMixture` (EM), diagonal covariance, `n_init=3`
- **K = 7** (minimum AIC and BIC on K = 2…8)
- Converged in 32 EM iterations
- Average log-likelihood 19.335; AIC −49865; BIC −49111
- GMM silhouette −0.038 (overlapping mixtures); K-Means silhouette 0.180 at the same K
- Cluster sizes: 204, 139, 195, 96, 203, 336, 124

Silhouette is highest at **K = 2**; that measures hard-assignment separation, not mixture fit. See the report for the interpretation of each cluster from rating-share profiles.
