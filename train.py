"""
Run once: python train.py
Trains all models and saves to models/ directory.
Expected time: 5–10 minutes on CPU (custom ALS, no GPU).
"""

import time
import joblib
import pandas as pd
from pathlib import Path

DATA_DIR   = Path("data/ml-1m")
MODELS_DIR = Path("models")


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    # ── 1. Load raw data ───────────────────────────────────────────────────
    stamp("Loading data …")
    from utils.data_loader import DataLoader
    loader = DataLoader()

    ratings_df = loader.load_ratings(str(DATA_DIR / "ratings.dat"))
    movies_df  = loader.load_movies(str(DATA_DIR / "movies.dat"), ratings_df=ratings_df)
    users_df   = loader.load_users(str(DATA_DIR / "users.dat"))

    stamp(
        f"  Movies: {len(movies_df):,}  |  "
        f"Ratings: {len(ratings_df):,}  |  "
        f"Users: {len(users_df):,}"
    )

    stamp("Saving preprocessed DataFrames …")
    joblib.dump(movies_df,  MODELS_DIR / "movies.pkl")
    joblib.dump(ratings_df, MODELS_DIR / "ratings.pkl")
    joblib.dump(users_df,   MODELS_DIR / "users.pkl")

    # ── 2. Content-based ───────────────────────────────────────────────────
    stamp("Training ContentBasedRecommender …")
    from recommenders.content_based import ContentBasedRecommender
    cb = ContentBasedRecommender()
    cb.fit(movies_df)
    joblib.dump(cb, MODELS_DIR / "cb_model.pkl")
    stamp("  ✓ ContentBased saved.")

    # ── 3. Collaborative (ALS) ─────────────────────────────────────────────
    stamp("Training CollaborativeFilterRecommender (ALS) …")
    from recommenders.collaborative import CollaborativeFilterRecommender
    cf = CollaborativeFilterRecommender()
    cf.fit(ratings_df, movies_df=movies_df)
    joblib.dump(cf, MODELS_DIR / "cf_model.pkl")
    stamp("  ✓ Collaborative saved.")

    # ── 4. Hybrid ──────────────────────────────────────────────────────────
    stamp("Creating HybridRecommender …")
    from recommenders.hybrid import HybridRecommender
    hybrid = HybridRecommender(cb, cf)
    joblib.dump(hybrid, MODELS_DIR / "hybrid_model.pkl")
    stamp("  ✓ Hybrid saved.")

    # ── 5. Evaluation metrics ──────────────────────────────────────────────
    stamp("Computing evaluation metrics …")
    from recommenders.evaluator import MetricsEvaluator
    evaluator = MetricsEvaluator()

    stamp("  Computing RMSE / MAE (SVD) …")
    eval_results = evaluator.compute_rmse_mae(ratings_df)
    stamp(
        f"  SVD  → RMSE={eval_results['SVD_RMSE']:.4f}  MAE={eval_results['SVD_MAE']:.4f}\n"
        f"  Base → RMSE={eval_results['Baseline_RMSE']:.4f}  MAE={eval_results['Baseline_MAE']:.4f}"
    )

    stamp("  Computing Precision / NDCG for ContentBased (200 users) …")
    cb_metrics = evaluator.compute_ranking_metrics(cb, ratings_df, n_users=200)

    stamp("  Computing Precision / NDCG for Collaborative (200 users) …")
    cf_metrics = evaluator.compute_ranking_metrics(cf, ratings_df, n_users=200)

    # pd.concat instead of deprecated ._append
    frames = [df for df in [cb_metrics, cf_metrics] if not df.empty]
    all_metrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    joblib.dump(
        {"eval_results": eval_results, "ranking_metrics": all_metrics},
        MODELS_DIR / "metrics.pkl",
    )
    stamp("  ✓ Metrics saved.")

    # ── Done ───────────────────────────────────────────────────────────────
    stamp("=" * 50)
    stamp("All done!  Models saved to  models/")
    stamp("Run:  streamlit run app.py")


if __name__ == "__main__":
    main()
