from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

PRIMARY    = "#6C63FF"
SECONDARY  = "#FF6B6B"
SUCCESS    = "#4ECDC4"
CARD_BG    = "#1E2130"
TEXT_MAIN  = "#FAFAFA"
TEXT_MUTED = "#A0A8B8"
BG_DARK    = "#0E1117"

# Base layout WITHOUT xaxis/yaxis so callers can add their own without conflict
_BASE = dict(
    paper_bgcolor=BG_DARK,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT_MAIN, family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
)

_AXIS = dict(gridcolor="#2A2D3E", zerolinecolor="#2A2D3E")  # reusable axis style

GENRE_COLORS = [
    "#6C63FF", "#FF6B6B", "#4ECDC4", "#FFE66D", "#A8E6CF",
    "#FF8B94", "#B4A7D6", "#F9D423", "#FC913A", "#FF4E50",
    "#00B4DB", "#0083B0", "#BC4E9C", "#F80759", "#8360C3",
    "#2EBD59", "#11998E", "#38EF7D", "#F7971E", "#FFD200",
]


def _title(text: str) -> dict:
    return dict(text=text, font=dict(size=15, color=TEXT_MAIN))


# ── Rating distribution ────────────────────────────────────────────────────────
def plot_rating_distribution(ratings_df: pd.DataFrame) -> go.Figure:
    counts = ratings_df["rating"].value_counts().sort_index()
    labels = [str(int(r)) if r == int(r) else str(r) for r in counts.index]

    fig = go.Figure(go.Bar(
        x=labels,
        y=counts.values,
        marker=dict(
            color=counts.values,
            colorscale=[[0, PRIMARY], [1, SECONDARY]],
            showscale=False,
        ),
        text=counts.values,
        textposition="outside",
        textfont=dict(color=TEXT_MUTED, size=11),
    ))
    fig.update_layout(
        **_BASE,
        title=_title("Розподіл оцінок"),
        xaxis=dict(**_AXIS, title="Оцінка"),
        yaxis=dict(**_AXIS, title="Кількість"),
        height=350,
    )
    return fig


# ── Genre distribution ─────────────────────────────────────────────────────────
def plot_genre_distribution(movies_df: pd.DataFrame) -> go.Figure:
    genres = movies_df["genres"].str.split("|").explode()
    counts = genres.value_counts().head(15)
    n = len(counts)
    colors = [GENRE_COLORS[i % len(GENRE_COLORS)] for i in range(n)]

    fig = go.Figure(go.Bar(
        x=counts.values,
        y=counts.index,
        orientation="h",
        marker=dict(color=colors),
        text=counts.values,
        textposition="outside",
    ))
    fig.update_layout(
        **_BASE,
        title=_title("Топ-15 жанрів"),
        xaxis=dict(**_AXIS, title="Кількість фільмів"),
        yaxis=dict(**_AXIS, autorange="reversed"),
        height=450,
    )
    return fig


# ── TF-IDF heatmap ────────────────────────────────────────────────────────────
def plot_tfidf_heatmap(cb_model, movies_df: pd.DataFrame) -> go.Figure:
    top = (
        movies_df.nlargest(20, "n_ratings")
        if "n_ratings" in movies_df.columns
        else movies_df.head(20)
    )
    titles  = [t[:25] + "…" if len(t) > 25 else t for t in top["title"].values]
    indices = [cb_model.movie_idx.get(int(mid)) for mid in top["movieId"]]
    valid   = [(t, i) for t, i in zip(titles, indices) if i is not None]
    if not valid:
        return go.Figure()
    titles_v, indices_v = zip(*valid)
    matrix = cb_model.get_tfidf_matrix()[list(indices_v), :]
    feature_names = cb_model.vectorizer.get_feature_names_out()

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=feature_names,
        y=list(titles_v),
        colorscale="Blues",
        colorbar=dict(tickfont=dict(color=TEXT_MUTED)),
    ))
    fig.update_layout(
        **_BASE,
        title=_title("TF-IDF матриця жанрів (топ-20 фільмів)"),
        xaxis=dict(**_AXIS, tickangle=-45),
        yaxis=dict(**_AXIS),
        height=550,
    )
    return fig


# ── Cosine similarity matrix ───────────────────────────────────────────────────
def plot_cosine_similarity(cb_model, movies_df: pd.DataFrame) -> go.Figure:
    from sklearn.metrics.pairwise import cosine_similarity

    top = (
        movies_df.nlargest(15, "n_ratings")
        if "n_ratings" in movies_df.columns
        else movies_df.head(15)
    )
    titles  = [t[:20] + "…" if len(t) > 20 else t for t in top["title"].values]
    indices = [cb_model.movie_idx.get(int(mid)) for mid in top["movieId"]]
    valid   = [(t, i) for t, i in zip(titles, indices) if i is not None]
    if not valid:
        return go.Figure()
    titles_v, indices_v = zip(*valid)
    matrix = cb_model.get_tfidf_matrix()[list(indices_v), :]
    sim = cosine_similarity(matrix)

    annotations = [
        dict(x=j, y=i, text=f"{sim[i, j]:.2f}",
             showarrow=False, font=dict(size=9, color="white"))
        for i in range(len(titles_v))
        for j in range(len(titles_v))
    ]

    fig = go.Figure(go.Heatmap(
        z=sim,
        x=list(titles_v),
        y=list(titles_v),
        colorscale=[[0, CARD_BG], [0.5, PRIMARY], [1, SECONDARY]],
        zmin=0, zmax=1,
        colorbar=dict(tickfont=dict(color=TEXT_MUTED)),
    ))
    fig.update_layout(
        **_BASE,
        title=_title("Косинусна схожість (топ-15 фільмів)"),
        xaxis=dict(**_AXIS, tickangle=-45),
        yaxis=dict(**_AXIS),
        height=550,
        annotations=annotations,
    )
    return fig


# ── Confidence formula ────────────────────────────────────────────────────────
def plot_confidence_formula() -> go.Figure:
    r = np.linspace(0, 5, 200)
    alphas = [10, 20, 40, 80]
    colors = [SUCCESS, PRIMARY, SECONDARY, "#FFE66D"]

    fig = go.Figure()
    for alpha, color in zip(alphas, colors):
        fig.add_trace(go.Scatter(
            x=r, y=1 + alpha * r,
            mode="lines",
            name=f"α = {alpha}",
            line=dict(color=color, width=2.5),
        ))
    for r_int in range(1, 6):
        for alpha, color in zip(alphas, colors):
            fig.add_trace(go.Scatter(
                x=[r_int], y=[1 + alpha * r_int],
                mode="markers",
                marker=dict(color=color, size=7),
                showlegend=False,
            ))

    fig.update_layout(
        **_BASE,
        title=_title("Confidence formula: c(u,i) = 1 + α·r(u,i)"),
        xaxis=dict(**_AXIS, title="Rating r(u,i)"),
        yaxis=dict(**_AXIS, title="Confidence c(u,i)"),
        height=400,
        legend=dict(bgcolor=CARD_BG, bordercolor="#2A2D3E"),
    )
    return fig


# ── Latent space (PCA) ────────────────────────────────────────────────────────
def plot_latent_space(cf_model, movies_df: pd.DataFrame) -> go.Figure:
    item_factors = cf_model.get_item_factors()
    if item_factors is None or len(item_factors) < 10:
        return go.Figure()

    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(item_factors)

    movie_ids  = cf_model.le_movie.classes_
    id_to_info = movies_df.set_index("movieId")[["title", "genres"]].to_dict("index")

    rows = []
    for i, mid in enumerate(movie_ids):
        info = id_to_info.get(mid, {})
        rows.append({
            "x":     float(coords[i, 0]),
            "y":     float(coords[i, 1]),
            "title": str(info.get("title", str(mid)))[:30],
            "genre": str(info.get("genres", "Unknown")).split("|")[0],
        })
    df_plot = pd.DataFrame(rows)

    unique_genres = df_plot["genre"].unique()
    color_map = {g: GENRE_COLORS[i % len(GENRE_COLORS)] for i, g in enumerate(unique_genres)}

    fig = go.Figure()
    for genre, grp in df_plot.groupby("genre"):
        fig.add_trace(go.Scatter(
            x=grp["x"], y=grp["y"],
            mode="markers",
            name=genre,
            marker=dict(color=color_map[genre], size=5, opacity=0.7),
            text=grp["title"],
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        **_BASE,
        title=_title("Латентний простір ALS (PCA 2D)"),
        xaxis=dict(**_AXIS, title="PC 1"),
        yaxis=dict(**_AXIS, title="PC 2"),
        height=520,
        legend=dict(bgcolor=CARD_BG, bordercolor="#2A2D3E", font=dict(size=10)),
    )
    return fig


# ── ALS training loss ─────────────────────────────────────────────────────────
def plot_als_training_loss(cf_model) -> go.Figure:
    loss = cf_model.get_training_loss()
    if not loss:
        return go.Figure()

    fig = go.Figure(go.Scatter(
        x=list(range(1, len(loss) + 1)),
        y=loss,
        mode="lines+markers",
        line=dict(color=PRIMARY, width=2.5),
        marker=dict(color=SECONDARY, size=7),
        name="Training Loss",
    ))
    fig.update_layout(
        **_BASE,
        title=_title("ALS: втрата при навчанні"),
        xaxis=dict(**_AXIS, title="Ітерація"),
        yaxis=dict(**_AXIS, title="Loss"),
        height=350,
    )
    return fig


# ── Precision / NDCG comparison ───────────────────────────────────────────────
def plot_metrics_comparison(metrics_df: pd.DataFrame) -> go.Figure:
    if metrics_df.empty:
        return go.Figure()

    methods = metrics_df["Method"].unique()
    colors  = [PRIMARY, SECONDARY, SUCCESS, "#FFE66D"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Precision@K", "NDCG@K"],
    )
    for idx, method in enumerate(methods):
        sub   = metrics_df[metrics_df["Method"] == method]
        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Bar(x=sub["K"].astype(str), y=sub["Precision"],
                   name=method, marker_color=color, legendgroup=method),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=sub["K"], y=sub["NDCG"],
                       mode="lines+markers", name=method,
                       line=dict(color=color, width=2), marker=dict(size=8),
                       legendgroup=method, showlegend=False),
            row=1, col=2,
        )

    fig.update_layout(
        **_BASE,
        height=380,
        barmode="group",
        legend=dict(bgcolor=CARD_BG, bordercolor="#2A2D3E"),
    )
    fig.update_xaxes(gridcolor="#2A2D3E", zerolinecolor="#2A2D3E")
    fig.update_yaxes(gridcolor="#2A2D3E", zerolinecolor="#2A2D3E")
    return fig


# ── RMSE / MAE bar chart ──────────────────────────────────────────────────────
def plot_rmse_mae(eval_results: dict) -> go.Figure:
    models    = ["Baseline", "SVD"]
    rmse_vals = [eval_results.get("Baseline_RMSE", 0), eval_results.get("SVD_RMSE", 0)]
    mae_vals  = [eval_results.get("Baseline_MAE",  0), eval_results.get("SVD_MAE",  0)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="RMSE", x=models, y=rmse_vals,
        marker_color=SECONDARY,
        text=[f"{v:.4f}" for v in rmse_vals], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="MAE", x=models, y=mae_vals,
        marker_color=SUCCESS,
        text=[f"{v:.4f}" for v in mae_vals], textposition="outside",
    ))
    fig.update_layout(
        **_BASE,
        title=_title("RMSE та MAE: Baseline vs SVD"),
        xaxis=dict(**_AXIS),
        yaxis=dict(**_AXIS, title="Error"),
        barmode="group",
        height=380,
        legend=dict(bgcolor=CARD_BG, bordercolor="#2A2D3E"),
    )
    return fig
