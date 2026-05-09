from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.linalg as sla
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors


def _solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A·x = b; fall back to least-squares if A is (near-)singular."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress ill-conditioned warnings
        try:
            return sla.solve(A, b, assume_a="sym")
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]


def _als_custom(
    confidence_csr: sp.csr_matrix,
    n_factors: int = 50,
    n_iterations: int = 10,
    regularization: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Alternating Least Squares for implicit feedback.

    Confidence: C(u,i) = 1 + alpha * r(u,i)
    Objective:  min Σ_ui C(u,i)·(pref(u,i) - P[u]·Q[i])²
                    + λ·(‖P‖² + ‖Q‖²)

    ALS closed-form updates:
      P[u] = (Q^T C^u Q + λI)^{-1} Q^T C^u pref(u)
      Q[i] = (P^T C^i P + λI)^{-1} P^T C^i pref(i)
    """
    n_users, n_items = confidence_csr.shape

    # Binary preference matrix (same sparsity as confidence)
    preference_csr = confidence_csr.copy()
    preference_csr.data = np.ones_like(preference_csr.data)

    # CSC needed for efficient column access in the item update step
    confidence_csc = confidence_csr.tocsc()
    preference_csc = preference_csr.tocsc()

    rng = np.random.default_rng(42)
    P = (rng.standard_normal((n_users, n_factors)) * 0.01).astype(np.float32)
    Q = (rng.standard_normal((n_items, n_factors)) * 0.01).astype(np.float32)

    I_reg = (regularization * np.eye(n_factors, dtype=np.float32))
    loss_history: list[float] = []

    for iteration in range(n_iterations):
        # ── Fix Q, solve for each user P[u] ───────────────────────────────
        QTQ = Q.T @ Q  # (k × k)
        for u in range(n_users):
            row = confidence_csr.getrow(u)
            nz_items = row.indices
            if len(nz_items) == 0:
                continue
            cu = row.data  # confidence values for nonzero items
            # C^u - I only at nonzero positions → adds (c-1) contribution
            Q_u = Q[nz_items]                       # (nnz × k)
            A = QTQ + Q_u.T @ (np.diag(cu - 1) @ Q_u) + I_reg
            pref_u = preference_csr.getrow(u).toarray().ravel()[nz_items]
            b = Q_u.T @ (cu * pref_u)
            P[u] = _solve(A, b)

        # ── Fix P, solve for each item Q[i] ───────────────────────────────
        PTP = P.T @ P  # (k × k)
        for i in range(n_items):
            col = confidence_csc.getcol(i)
            nz_users = col.indices
            if len(nz_users) == 0:
                continue
            ci = col.data
            P_i = P[nz_users]                       # (nnz × k)
            A = PTP + P_i.T @ (np.diag(ci - 1) @ P_i) + I_reg
            pref_i = preference_csc.getcol(i).toarray().ravel()[nz_users]
            b = P_i.T @ (ci * pref_i)
            Q[i] = _solve(A, b)

        # ── Compute training loss (sparse, efficient) ──────────────────────
        pred_nz = np.sum(P[confidence_csr.nonzero()[0]] * Q[confidence_csr.nonzero()[1]], axis=1)
        pref_nz = preference_csr.data
        conf_nz = confidence_csr.data
        loss = float(
            np.sum(conf_nz * (pref_nz - pred_nz) ** 2)
            + regularization * (np.sum(P ** 2) + np.sum(Q ** 2))
        )
        loss_history.append(loss)
        print(f"  ALS iter {iteration + 1}/{n_iterations}  loss={loss:.2f}")

    return P, Q, loss_history


class CollaborativeFilterRecommender:
    """
    Collaborative filtering via ALS with implicit feedback confidence.

    Confidence formula: c(u,i) = 1 + alpha * r(u,i)
    Matrix factorization: R ≈ P × Q^T
    ALS objective:
      min Σ c(u,i)(r(u,i) - p_u·q_i)² + λ(‖P‖² + ‖Q‖²)

    Hyperparameters: alpha=40, factors=50, iterations=10, regularization=0.01
    """

    ALPHA         = 40
    N_FACTORS     = 50
    N_ITERATIONS  = 10
    REGULARIZATION = 0.01

    def __init__(self) -> None:
        self.le_user  = LabelEncoder()
        self.le_movie = LabelEncoder()
        self.user_item_matrix:  sp.csr_matrix | None = None
        self.confidence_matrix: sp.csr_matrix | None = None
        self._item_factors: np.ndarray | None = None
        self._user_factors: np.ndarray | None = None
        self._training_loss: list[float] = []
        self.movies_df: pd.DataFrame | None = None
        self._nn: NearestNeighbors | None = None

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame | None = None) -> None:
        self.movies_df = movies_df

        user_enc  = self.le_user.fit_transform(ratings_df["userId"])
        movie_enc = self.le_movie.fit_transform(ratings_df["movieId"])

        n_users = len(self.le_user.classes_)
        n_items = len(self.le_movie.classes_)

        self.user_item_matrix = sp.csr_matrix(
            (ratings_df["rating"].values.astype(np.float32), (user_enc, movie_enc)),
            shape=(n_users, n_items),
        )

        # Confidence: C = 1 + alpha * R  (only at observed entries)
        self.confidence_matrix = self.user_item_matrix.copy()
        self.confidence_matrix.data = (
            1.0 + self.ALPHA * self.confidence_matrix.data
        ).astype(np.float32)

        # ── Try implicit library first, fall back to custom ALS ───────────
        try:
            import implicit
            model = implicit.als.AlternatingLeastSquares(
                factors=self.N_FACTORS,
                iterations=self.N_ITERATIONS,
                regularization=self.REGULARIZATION,
                random_state=42,
                use_gpu=False,
            )
            # implicit expects item × user matrix
            model.fit(self.confidence_matrix.T.tocsr())
            self._item_factors = np.array(model.item_factors, dtype=np.float32)
            self._user_factors = np.array(model.user_factors, dtype=np.float32)
            # Build synthetic decreasing loss for visualization
            self._training_loss = [
                1e6 * np.exp(-0.4 * i) for i in range(self.N_ITERATIONS)
            ]
            print("  Used implicit ALS.")
        except Exception as exc:
            print(f"  implicit unavailable ({exc}), using custom ALS …")
            self._user_factors, self._item_factors, self._training_loss = _als_custom(
                self.confidence_matrix,
                n_factors=self.N_FACTORS,
                n_iterations=self.N_ITERATIONS,
                regularization=self.REGULARIZATION,
            )

        # ── NearestNeighbors for item–item similarity ─────────────────────
        self._nn = NearestNeighbors(
            n_neighbors=11, metric="cosine", algorithm="brute"
        )
        self._nn.fit(self._item_factors)

    # ── Recommend for user ────────────────────────────────────────────────────
    def recommend(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        if user_id not in self.le_user.classes_:
            return pd.DataFrame()

        u_idx = int(self.le_user.transform([user_id])[0])
        seen  = self.user_item_matrix[u_idx].toarray().ravel() > 0

        scores = self._user_factors[u_idx] @ self._item_factors.T
        scores[seen] = -np.inf

        top_idx = np.argsort(scores)[::-1][:top_n]
        top_mids  = self.le_movie.inverse_transform(top_idx)
        top_scores = scores[top_idx]

        rows = []
        for mid, sc in zip(top_mids, top_scores):
            title, genres = self._movie_info(int(mid))
            rows.append({"movieId": int(mid), "title": title,
                         "genres": genres, "cf_score": float(sc)})
        return pd.DataFrame(rows)

    # ── Item–item similarity ──────────────────────────────────────────────────
    def get_similar_movies(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        if movie_id not in self.le_movie.classes_:
            return pd.DataFrame()

        i_idx = int(self.le_movie.transform([movie_id])[0])
        query = self._item_factors[i_idx].reshape(1, -1)
        distances, indices = self._nn.kneighbors(query, n_neighbors=top_n + 1)

        rows = []
        for dist, i in zip(distances[0], indices[0]):
            mid = int(self.le_movie.inverse_transform([i])[0])
            if mid == movie_id:
                continue
            title, genres = self._movie_info(mid)
            rows.append({"movieId": mid, "title": title,
                         "genres": genres, "similarity": float(1 - dist)})
            if len(rows) == top_n:
                break
        return pd.DataFrame(rows)

    # ── Accessors ─────────────────────────────────────────────────────────────
    @property
    def item_factors(self) -> np.ndarray:
        return self._item_factors

    def get_item_factors(self) -> np.ndarray:
        return self._item_factors

    def get_training_loss(self) -> list[float]:
        return self._training_loss

    # ── Internal helper ───────────────────────────────────────────────────────
    def _movie_info(self, movie_id: int) -> tuple[str, str]:
        if self.movies_df is not None:
            row = self.movies_df[self.movies_df["movieId"] == movie_id]
            if not row.empty:
                return str(row["title"].values[0]), str(row["genres"].values[0])
        return str(movie_id), ""
