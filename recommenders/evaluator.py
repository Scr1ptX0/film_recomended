from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds


class MetricsEvaluator:
    """
    Evaluation metrics:
    - RMSE = sqrt(mean((y_true - y_pred)²))
    - MAE  = mean(|y_true - y_pred|)
    - Precision@K = |relevant ∩ recommended| / K
    - Recall@K    = |relevant ∩ recommended| / |relevant|
    - NDCG@K = DCG@K / IDCG@K   where DCG = Σ rel_i / log₂(i+2)
    """

    def __init__(self) -> None:
        self._train_df: pd.DataFrame | None = None
        self._test_df:  pd.DataFrame | None = None

    # ── Temporal train/test split ──────────────────────────────────────────
    def train_test_split_temporal(
        self, ratings_df: pd.DataFrame, test_ratio: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        sorted_df = ratings_df.sort_values("timestamp").reset_index(drop=True)
        split_idx = int(len(sorted_df) * (1 - test_ratio))
        train = sorted_df.iloc[:split_idx].copy()
        test  = sorted_df.iloc[split_idx:].copy()
        self._train_df = train
        self._test_df  = test
        return train, test

    # ── RMSE / MAE via SVD ─────────────────────────────────────────────────
    def compute_rmse_mae(self, ratings_df: pd.DataFrame) -> dict:
        train, test = self.train_test_split_temporal(ratings_df)

        all_users  = sorted(ratings_df["userId"].unique())
        all_movies = sorted(ratings_df["movieId"].unique())
        u2i = {u: i for i, u in enumerate(all_users)}
        m2i = {m: i for i, m in enumerate(all_movies)}

        # Build train sparse matrix
        train_u = train["userId"].map(u2i).values
        train_m = train["movieId"].map(m2i).values
        R = sp.csr_matrix(
            (train["rating"].values.astype(np.float64), (train_u, train_m)),
            shape=(len(all_users), len(all_movies)),
        )

        # SVD-based prediction
        k = min(50, min(R.shape) - 1)
        U, sigma, Vt = svds(R, k=k)
        pred_matrix = U @ np.diag(sigma) @ Vt  # dense (n_users × n_items)

        # Filter test to known users / movies
        test_u = test["userId"].map(u2i)
        test_m = test["movieId"].map(m2i)
        valid  = test_u.notna() & test_m.notna()
        test_v = test[valid]
        tu = test_u[valid].astype(int).values
        tm = test_m[valid].astype(int).values
        y_true = test_v["rating"].values.astype(float)

        y_pred_svd  = np.clip(pred_matrix[tu, tm], 1.0, 5.0)
        global_mean = float(train["rating"].mean())
        y_pred_base = np.full_like(y_true, global_mean)

        return {
            "SVD_RMSE":      float(np.sqrt(np.mean((y_true - y_pred_svd)  ** 2))),
            "SVD_MAE":       float(np.mean(np.abs(y_true  - y_pred_svd))),
            "Baseline_RMSE": float(np.sqrt(np.mean((y_true - y_pred_base) ** 2))),
            "Baseline_MAE":  float(np.mean(np.abs(y_true  - y_pred_base))),
        }

    # ── Ranking metrics ────────────────────────────────────────────────────
    def compute_ranking_metrics(
        self,
        recommender,
        ratings_df: pd.DataFrame,
        n_users: int = 200,
        k_vals: list[int] | None = None,
    ) -> pd.DataFrame:
        if k_vals is None:
            k_vals = [5, 10, 20]

        train, test = self.train_test_split_temporal(ratings_df)
        rel_threshold = 4.0

        # Relevant items per test user
        test_relevant: dict[int, set] = (
            test[test["rating"] >= rel_threshold]
            .groupby("userId")["movieId"]
            .apply(set)
            .to_dict()
        )

        eligible = [u for u, rel in test_relevant.items() if len(rel) > 0]
        rng      = np.random.default_rng(42)
        sample   = rng.choice(
            eligible, size=min(n_users, len(eligible)), replace=False
        )

        is_cb  = hasattr(recommender, "recommend_by_user")
        method = recommender.__class__.__name__
        records: list[dict] = []

        for k in k_vals:
            prec_l, rec_l, ndcg_l = [], [], []
            for uid in sample:
                relevant = test_relevant[uid]
                try:
                    if is_cb:
                        recs_df = recommender.recommend_by_user(
                            int(uid), train, top_n=k
                        )
                    else:
                        recs_df = recommender.recommend(int(uid), top_n=k)

                    if recs_df is None or recs_df.empty:
                        continue
                    rec_ids = list(recs_df["movieId"].values[:k])
                except Exception:
                    continue

                hits  = [1 if mid in relevant else 0 for mid in rec_ids]
                n_hit = sum(hits)

                prec = n_hit / k
                rec  = n_hit / len(relevant)

                dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
                idcg = sum(1 / np.log2(i + 2) for i in range(min(n_hit, k)))
                ndcg = dcg / idcg if idcg > 0 else 0.0

                prec_l.append(prec)
                rec_l.append(rec)
                ndcg_l.append(ndcg)

            if prec_l:
                records.append({
                    "Method":    method,
                    "K":         k,
                    "Precision": float(np.mean(prec_l)),
                    "Recall":    float(np.mean(rec_l)),
                    "NDCG":      float(np.mean(ndcg_l)),
                })

        return pd.DataFrame(records)
