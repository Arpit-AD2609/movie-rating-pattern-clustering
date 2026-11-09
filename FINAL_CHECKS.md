# Final Project Checks

## Verified in the project workspace

- Python compilation: `python -m compileall -q src app.py` passed.
- Reference pipeline: `run_analysis(min_ratings=20, k=7, seed=42)` completed.
- Movies retained: 1,297.
- GMM silhouette: -0.037884636950795386 (approximately -0.038).
- K-Means silhouette: 0.180032679292599 (approximately 0.180).
- GMM convergence: True; iterations: 32.
- GMM cluster sizes: 204, 139, 195, 96, 203, 336, 124.
- Plotly `cluster_profiles` and `pca_scatter` functions returned valid Plotly Figure objects, with 7 traces each.

## Not verified in this environment

- Streamlit is listed in `requirements.txt` but is not installed in the packaging environment, so the live Streamlit server and browser-rendered UI were not verified here.
- Please install dependencies and launch the app locally using the commands in `README.md`, then click through each page once before submission.
