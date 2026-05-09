from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


class ContentBasedRecommender:
    """
    Content-based filtering using TF-IDF on movie genres.

    TF-IDF formula: W(i,j) = TF(i,j) × log(N / DF(i))
    Similarity:     cosine(q, d) = (q·d) / (‖q‖ × ‖d‖)
    """

    def __init__(self) -> None:
        self.vectorizer:   TfidfVectorizer | None = None
        self.tfidf_matrix: np.ndarray | None = None
        self.movie_idx:    dict[int, int]    = {}   # movieId → row index
        self.movies_df:    pd.DataFrame | None = None
        self._nn:          NearestNeighbors | None = None

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, movies_df: pd.DataFrame) -> None:
        self.movies_df = movies_df.copy().reset_index(drop=True)
        self.movie_idx = {
            int(mid): idx
            for idx, mid in enumerate(self.movies_df["movieId"])
        }

        self.vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+")
        sparse = self.vectorizer.fit_transform(self.movies_df["genres_str"])
        self.tfidf_matrix = normalize(sparse, norm="l2").toarray().astype(np.float32)

        n_neighbors = min(21, len(self.movies_df))
        self._nn = NearestNeighbors(
            n_neighbors=n_neighbors, metric="cosine", algorithm="brute"
        )
        self._nn.fit(self.tfidf_matrix)

    # ── Similar movies by movie_id ────────────────────────────────────────────
    def recommend_by_movie(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        idx = self.movie_idx.get(int(movie_id))
        if idx is None:
            return pd.DataFrame()

        query = self.tfidf_matrix[idx].reshape(1, -1)
        k = min(top_n + 1, len(self.movies_df))
        distances, indices = self._nn.kneighbors(query, n_neighbors=k)

        rows: list[dict] = []
        for dist, i in zip(distances[0], indices[0]):
            row = self.movies_df.iloc[i]
            mid = int(row["movieId"])
            if mid == int(movie_id):
                continue
            rows.append({
                "movieId":          mid,
                "title":            row["title"],
                "genres":           row["genres"],
                "similarity_score": float(1.0 - dist),
            })
            if len(rows) == top_n:
                break
        return pd.DataFrame(rows)

    # ── Personalised recommendations for a user ───────────────────────────────
    def recommend_by_user(
        self,
        user_id: int,
        ratings_df: pd.DataFrame,
        top_n: int = 10,
    ) -> pd.DataFrame:
        user_ratings = ratings_df[ratings_df["userId"] == int(user_id)]
        if user_ratings.empty:
            return pd.DataFrame()

        seen_ids = set(user_ratings["movieId"].astype(int))

        # Keep only movies that exist in the CB index
        valid_mask = user_ratings["movieId"].astype(int).isin(self.movie_idx)
        user_ratings = user_ratings[valid_mask]
        if user_ratings.empty:
            return pd.DataFrame()

        weights = user_ratings["rating"].values.astype(np.float32)
        weights /= weights.sum()

        idx_list = [self.movie_idx[int(m)] for m in user_ratings["movieId"]]
        profile  = np.average(self.tfidf_matrix[idx_list], axis=0, weights=weights)
        profile  = normalize(profile.reshape(1, -1), norm="l2")

        k = min(top_n * 4, len(self.movies_df))
        distances, indices = self._nn.kneighbors(profile, n_neighbors=k)

        rows: list[dict] = []
        for dist, i in zip(distances[0], indices[0]):
            row = self.movies_df.iloc[i]
            mid = int(row["movieId"])
            if mid in seen_ids:
                continue
            rows.append({
                "movieId":  mid,
                "title":    row["title"],
                "genres":   row["genres"],
                "cb_score": float(1.0 - dist),
            })
            if len(rows) == top_n:
                break
        return pd.DataFrame(rows)

    # ── Accessors ─────────────────────────────────────────────────────────────
    def get_genre_vector(self, movie_id: int) -> np.ndarray:
        idx = self.movie_idx.get(int(movie_id))
        if idx is None:
            return np.zeros(self.tfidf_matrix.shape[1], dtype=np.float32)
        return self.tfidf_matrix[idx]

    def get_tfidf_matrix(self) -> np.ndarray:
        return self.tfidf_matrix
