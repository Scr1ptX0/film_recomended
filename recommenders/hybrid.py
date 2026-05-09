from __future__ import annotations

import numpy as np
import pandas as pd

from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeFilterRecommender


class HybridRecommender:
    """
    Weighted hybrid: score = alpha * cb_score + (1 - alpha) * cf_score

    Adaptive alpha based on user activity:
    - < 10 ratings  → alpha = 0.80  (cold start: rely on content)
    - 10-50         → alpha = 0.60
    - 50-200        → alpha = 0.40
    - > 200 ratings → alpha = 0.20  (active user: rely on CF)
    """

    def __init__(
        self,
        cb: ContentBasedRecommender,
        cf: CollaborativeFilterRecommender,
    ) -> None:
        self.cb = cb
        self.cf = cf

    def __repr__(self) -> str:
        return f"HybridRecommender(cb={self.cb!r}, cf={self.cf!r})"

    def get_alpha(self, user_id: int, ratings_df: pd.DataFrame) -> float:
        n = int((ratings_df["userId"] == user_id).sum())
        if n < 10:
            return 0.80
        elif n < 50:
            return 0.60
        elif n < 200:
            return 0.40
        return 0.20

    def recommend(
        self,
        user_id: int,
        ratings_df: pd.DataFrame,
        top_n: int = 10,
    ) -> pd.DataFrame:
        alpha = self.get_alpha(user_id, ratings_df)
        fetch = top_n * 3

        cb_recs = self.cb.recommend_by_user(user_id, ratings_df, top_n=fetch)
        cf_recs = self.cf.recommend(user_id, top_n=fetch)

        # Normalize to [0, 1]
        def _norm(df: pd.DataFrame, col: str) -> pd.DataFrame:
            if df.empty:
                return df
            mn, mx = df[col].min(), df[col].max()
            if mx == mn:
                df[col] = 1.0
            else:
                df[col] = (df[col] - mn) / (mx - mn)
            return df

        cb_recs = _norm(cb_recs, "cb_score") if not cb_recs.empty else cb_recs
        cf_recs = _norm(cf_recs, "cf_score") if not cf_recs.empty else cf_recs

        if cb_recs.empty and cf_recs.empty:
            return pd.DataFrame()

        if cb_recs.empty:
            cf_recs["cb_score"] = 0.0
            cf_recs["hybrid_score"] = (1 - alpha) * cf_recs["cf_score"]
            cf_recs["alpha_used"] = alpha
            return cf_recs.head(top_n)

        if cf_recs.empty:
            cb_recs["cf_score"] = 0.0
            cb_recs["hybrid_score"] = alpha * cb_recs["cb_score"]
            cb_recs["alpha_used"] = alpha
            return cb_recs.head(top_n)

        merged = pd.merge(
            cb_recs[["movieId", "title", "genres", "cb_score"]],
            cf_recs[["movieId", "cf_score"]],
            on="movieId",
            how="outer",
        )
        merged["cb_score"] = merged["cb_score"].fillna(0.0)
        merged["cf_score"] = merged["cf_score"].fillna(0.0)

        # Fill title/genres from CF side if missing
        if "title" in cf_recs.columns:
            title_map = cf_recs.set_index("movieId")["title"].to_dict()
            genre_map = cf_recs.set_index("movieId")["genres"].to_dict()
            merged["title"] = merged.apply(
                lambda r: r["title"] if pd.notna(r["title"]) else title_map.get(r["movieId"], ""),
                axis=1,
            )
            merged["genres"] = merged.apply(
                lambda r: r["genres"] if pd.notna(r["genres"]) else genre_map.get(r["movieId"], ""),
                axis=1,
            )

        merged["hybrid_score"] = alpha * merged["cb_score"] + (1 - alpha) * merged["cf_score"]
        merged["alpha_used"] = alpha
        merged = merged.sort_values("hybrid_score", ascending=False).reset_index(drop=True)

        return merged.head(top_n)
