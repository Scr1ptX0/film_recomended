"""
Film Recommendation System — Streamlit App
Algorithmization and Programming (Data Science), university coursework.
Run: streamlit run app.py
"""

import re
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FilmRecommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Colour palette ───────────────────────────────────────────────────────────
PRIMARY    = "#6C63FF"
SECONDARY  = "#FF6B6B"
SUCCESS    = "#4ECDC4"
BG_DARK    = "#0E1117"
CARD_BG    = "#1E2130"
TEXT_MAIN  = "#FAFAFA"
TEXT_MUTED = "#A0A8B8"

# ─── Global CSS ──────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
::-webkit-scrollbar{{width:6px}}
::-webkit-scrollbar-track{{background:{BG_DARK}}}
::-webkit-scrollbar-thumb{{background:{PRIMARY};border-radius:3px}}

section[data-testid="stSidebar"]{{
  background:{CARD_BG}!important;
  border-right:1px solid #2A2D3E;
}}

/* metric card */
.metric-card{{
  background:{CARD_BG};border-radius:16px;padding:20px;text-align:center;
  border:1px solid #2A2D3E;box-shadow:0 4px 20px rgba(108,99,255,.1);
  min-height:110px;display:flex;flex-direction:column;justify-content:center;
}}
.metric-value{{
  font-size:2rem;font-weight:800;
  background:linear-gradient(135deg,{PRIMARY},{SECONDARY});
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}}
.metric-label{{color:{TEXT_MUTED};font-size:13px;margin-top:6px}}

/* page banner */
.page-banner{{
  background:linear-gradient(135deg,{PRIMARY} 0%,{SECONDARY} 100%);
  border-radius:16px;padding:36px 40px;margin-bottom:28px;
  box-shadow:0 8px 32px rgba(108,99,255,.25);
}}
.page-banner h1{{color:#fff!important;font-size:2.2rem;font-weight:800;margin:0 0 8px 0}}
.page-banner p{{color:rgba(255,255,255,.85);font-size:1rem;margin:0}}

/* algo card */
.algo-card{{
  background:{CARD_BG};border-radius:16px;padding:22px;
  border:1px solid #2A2D3E;box-shadow:0 4px 16px rgba(0,0,0,.2);height:100%;
}}
.algo-icon{{font-size:34px;margin-bottom:10px}}
.algo-title{{font-size:15px;font-weight:700;color:{TEXT_MAIN};margin-bottom:8px}}
.algo-desc{{font-size:13px;color:{TEXT_MUTED};line-height:1.65}}

/* recommendation card */
.rec-card{{
  background:{CARD_BG};border-radius:12px;padding:16px;margin:6px 0;
  border-left:4px solid {PRIMARY};box-shadow:0 4px 12px rgba(0,0,0,.3);
  transition:transform .15s;
}}
.rec-card:hover{{transform:translateX(4px)}}
.rec-title{{font-size:15px;font-weight:700;color:{TEXT_MAIN}}}
.rec-meta{{font-size:12px;color:{TEXT_MUTED};margin:4px 0}}
.score-bar-bg{{background:#2A2D3E;border-radius:8px;height:6px;margin:8px 0}}
.score-bar-fill{{
  background:linear-gradient(90deg,{PRIMARY},{SECONDARY});
  height:100%;border-radius:8px;
}}
.score-badge{{font-size:12px;color:{PRIMARY};font-weight:600}}

/* genre chip */
.genre-chip{{
  background:#2A2D3E;color:{PRIMARY};border:1px solid {PRIMARY};
  border-radius:20px;padding:2px 10px;font-size:11px;
  margin:2px;display:inline-block;
}}

/* search card */
.search-card{{
  background:{CARD_BG};border-radius:14px;padding:18px;margin-bottom:12px;
  border:1px solid #2A2D3E;box-shadow:0 2px 8px rgba(0,0,0,.2);
}}
.search-title{{font-size:15px;font-weight:700;color:{TEXT_MAIN};margin-bottom:4px}}
.year-badge{{
  background:{PRIMARY};color:#fff;border-radius:8px;
  padding:2px 8px;font-size:11px;font-weight:600;margin-left:6px;
}}

/* divider */
.section-divider{{height:1px;background:linear-gradient(90deg,{PRIMARY},transparent);margin:22px 0}}

/* tab styling */
.stTabs [data-baseweb="tab-list"]{{background:{CARD_BG};border-radius:10px 10px 0 0;padding:4px;gap:2px}}
.stTabs [data-baseweb="tab"]{{color:{TEXT_MUTED};border-radius:8px;padding:8px 16px}}
.stTabs [aria-selected="true"]{{background:{PRIMARY}!important;color:#fff!important}}
</style>
""", unsafe_allow_html=True)

MODELS_DIR = Path("models")


# ─── Model loading ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Завантаження моделей…")
def load_models():
    import joblib
    cb      = joblib.load(MODELS_DIR / "cb_model.pkl")
    cf      = joblib.load(MODELS_DIR / "cf_model.pkl")
    hybrid  = joblib.load(MODELS_DIR / "hybrid_model.pkl")
    movies  = joblib.load(MODELS_DIR / "movies.pkl")
    ratings = joblib.load(MODELS_DIR / "ratings.pkl")
    users   = joblib.load(MODELS_DIR / "users.pkl") if (MODELS_DIR / "users.pkl").exists() else pd.DataFrame()
    metrics = joblib.load(MODELS_DIR / "metrics.pkl") if (MODELS_DIR / "metrics.pkl").exists() else {}
    return cb, cf, hybrid, movies, ratings, users, metrics


def models_ready() -> bool:
    return MODELS_DIR.exists() and (MODELS_DIR / "cb_model.pkl").exists()


# ─── HTML helpers ─────────────────────────────────────────────────────────────
def _year(title: str) -> str:
    m = re.search(r"\((\d{4})\)$", title.strip())
    return m.group(1) if m else "—"


def genre_chips(genres: str) -> str:
    if not genres or genres in ("(no genres listed)", ""):
        return ""
    parts = genres.replace("|", " ").split()
    return "".join(f'<span class="genre-chip">{g}</span>' for g in parts)


def rec_card(title: str, year: str, genres: str, score_norm: float, score_raw: float, label: str = "Score") -> str:
    pct   = max(0.0, min(100.0, score_norm * 100))
    chips = genre_chips(genres)
    return f"""
<div class="rec-card">
  <div class="rec-title">{title}</div>
  <div class="rec-meta">{year}&nbsp;&nbsp;{chips}</div>
  <div class="score-bar-bg">
    <div class="score-bar-fill" style="width:{pct:.1f}%"></div>
  </div>
  <div class="score-badge">{label}: {score_raw:.4f}</div>
</div>"""


def metric_card(icon: str, value: str, label: str) -> str:
    return f"""
<div class="metric-card">
  <div style="font-size:24px">{icon}</div>
  <div class="metric-value">{value}</div>
  <div class="metric-label">{label}</div>
</div>"""


def star_rating(avg: float) -> str:
    full  = int(avg)
    half  = 1 if (avg - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + "½" * half + "☆" * empty + f"&nbsp;{avg:.1f}"


def _norm_series(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9) if mx != mn else pd.Series(np.ones(len(s)), index=s.index)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:16px 0 24px">
      <div style="font-size:42px">🎬</div>
      <div style="font-size:18px;font-weight:800;color:{TEXT_MAIN}">FilmRecommender</div>
      <div style="font-size:11px;color:{TEXT_MUTED};margin-top:4px">MovieLens 1M · University Project</div>
    </div>
    """, unsafe_allow_html=True)

    PAGE = st.radio(
        "nav",
        ["🏠 Головна", "🔍 Пошук фільмів", "👤 Рекомендації",
         "📊 Аналіз та метрики", "ℹ️ Про систему"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    if not models_ready():
        st.warning("Моделі відсутні.\nЗапустіть:\n```\npython train.py\n```")
    else:
        st.success("Моделі завантажено ✓")

# ─── Load models ──────────────────────────────────────────────────────────────
cb_model = cf_model = hybrid_model = movies_df = ratings_df = users_df = metrics = None

if models_ready():
    try:
        cb_model, cf_model, hybrid_model, movies_df, ratings_df, users_df, metrics = load_models()
    except Exception as e:
        st.error(f"Помилка завантаження моделей: {e}\nСпробуйте переnavчити: `python train.py`")
        st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Головна
# ═══════════════════════════════════════════════════════════════════════════════
if PAGE == "🏠 Головна":
    st.markdown("""
    <div class="page-banner">
      <h1>🎬 Система рекомендацій фільмів</h1>
      <p>Три алгоритми рекомендацій · MovieLens 1M · Курсова робота з алгоритмізації та програмування</p>
    </div>
    """, unsafe_allow_html=True)

    if not models_ready():
        st.warning("Спочатку навчіть моделі: `python train.py`")
        st.stop()

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, val, lbl in [
        (c1, "🎬", f"{len(movies_df):,}",        "Кількість фільмів"),
        (c2, "👥", f"{ratings_df['userId'].nunique():,}", "Кількість користувачів"),
        (c3, "⭐", f"{len(ratings_df):,}",        "Оцінок у датасеті"),
        (c4, "📊", f"{ratings_df['rating'].mean():.2f}", "Середня оцінка"),
    ]:
        col.markdown(metric_card(icon, val, lbl), unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:20px;font-weight:700;color:{TEXT_MAIN};margin-bottom:16px">Про систему</div>',
                unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)
    for col, icon, title, desc in [
        (a1, "📝", "Content-Based Filtering",
         "Аналізує жанри фільмів за допомогою TF-IDF. Будує профіль користувача "
         "та знаходить схожі фільми за косинусною схожістю."),
        (a2, "🤝", "Collaborative Filtering (ALS)",
         "Матрична факторизація з ALS та неявним зворотним зв'язком. "
         "c(u,i)=1+α·r(u,i). Знаходить латентні смаки користувачів."),
        (a3, "⚡", "Гібридний підхід",
         "Адаптивна комбінація CB та CF: score = α·cb + (1-α)·cf. "
         "α підбирається автоматично за кількістю оцінок користувача."),
    ]:
        col.markdown(f"""
        <div class="algo-card">
          <div class="algo-icon">{icon}</div>
          <div class="algo-title">{title}</div>
          <div class="algo-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    from utils.visualizations import plot_rating_distribution, plot_genre_distribution
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown(f'<div style="font-size:17px;font-weight:700;color:{TEXT_MAIN}">Розподіл оцінок</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(plot_rating_distribution(ratings_df), use_container_width=True)
    with ch2:
        st.markdown(f'<div style="font-size:17px;font-weight:700;color:{TEXT_MAIN}">Топ-15 жанрів</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(plot_genre_distribution(movies_df), use_container_width=True)

    st.markdown(f"""
    <div style="text-align:center;color:{TEXT_MUTED};font-size:12px;margin-top:32px;
                padding:16px;border-top:1px solid #2A2D3E;">
      F. Maxwell Harper and Joseph A. Konstan (2015). <i>The MovieLens Datasets: History and Context.</i>
      ACM TiiS 5, 4, Article 19.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Пошук фільмів
# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🔍 Пошук фільмів":
    st.markdown("""
    <div class="page-banner">
      <h1>🔍 Пошук фільмів</h1>
      <p>Шукайте фільми та переглядайте схожі рекомендації</p>
    </div>""", unsafe_allow_html=True)

    if not models_ready() or movies_df is None:
        st.warning("Спочатку навчіть моделі: `python train.py`")
        st.stop()

    query = st.text_input(
        "Введіть назву фільму…",
        placeholder="Наприклад: Star Wars, Toy Story, Matrix…",
        key="search_query",
    )

    if query.strip():
        found = movies_df[movies_df["title"].str.contains(query.strip(), case=False, na=False)].head(30)
        st.caption(f"Знайдено: {len(found)} фільм(ів)")
    else:
        found = (
            movies_df.nlargest(12, "n_ratings")
            if "n_ratings" in movies_df.columns
            else movies_df.head(12)
        )
        st.caption("Показано топ фільмів за кількістю оцінок")

    # Render cards 3 per row
    for row_slice in [found.iloc[i:i+3] for i in range(0, len(found), 3)]:
        cols = st.columns(3)
        for col, (_, mv) in zip(cols, row_slice.iterrows()):
            mid    = int(mv["movieId"])
            yr     = _year(str(mv["title"]))
            avg_r  = float(mv.get("avg_rating", 0))
            n_r    = int(mv.get("n_ratings", 0))
            chips  = genre_chips(str(mv.get("genres", "")))
            with col:
                st.markdown(f"""
                <div class="search-card">
                  <div class="search-title">
                    {mv['title']}<span class="year-badge">{yr}</span>
                  </div>
                  <div style="margin:6px 0">{chips}</div>
                  <div style="font-size:13px;color:{SUCCESS}">{star_rating(avg_r)}</div>
                  <div style="font-size:11px;color:{TEXT_MUTED}">{n_r:,} оцінок</div>
                </div>""", unsafe_allow_html=True)
                if st.button("🎬 Деталі та схожі", key=f"btn_{mid}"):
                    st.session_state["sel_movie"] = mid

    # Detail panel
    sel = st.session_state.get("sel_movie")
    if sel:
        row = movies_df[movies_df["movieId"] == sel]
        if not row.empty:
            mv = row.iloc[0]
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:20px;font-weight:700;color:{TEXT_MAIN}">'
                f'{mv["title"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(genre_chips(str(mv.get("genres", ""))), unsafe_allow_html=True)
            if "avg_rating" in mv:
                st.markdown(
                    f'<div style="color:{SUCCESS};font-size:14px;margin:6px 0">'
                    f'{star_rating(float(mv["avg_rating"]))} ({int(mv.get("n_ratings",0)):,} оцінок)</div>',
                    unsafe_allow_html=True,
                )

            tab_cb, tab_cf = st.tabs(["📝 Схожі за жанром (CB)", "🤝 Схожі за CF (ALS)"])

            with tab_cb:
                with st.spinner("Пошук схожих…"):
                    sim = cb_model.recommend_by_movie(sel, top_n=10)
                if sim.empty:
                    st.info("Схожих фільмів не знайдено.")
                else:
                    norms = _norm_series(sim["similarity_score"])
                    for (_, r), n in zip(sim.iterrows(), norms):
                        st.markdown(
                            rec_card(r["title"], _year(r["title"]),
                                     str(r.get("genres","")), float(n),
                                     float(r["similarity_score"]), "Схожість"),
                            unsafe_allow_html=True,
                        )

            with tab_cf:
                with st.spinner("Пошук схожих (ALS)…"):
                    sim2 = cf_model.get_similar_movies(sel, top_n=10)
                if sim2.empty:
                    st.info("Фільм відсутній у CF-моделі.")
                else:
                    norms2 = _norm_series(sim2["similarity"])
                    for (_, r), n in zip(sim2.iterrows(), norms2):
                        st.markdown(
                            rec_card(r["title"], _year(r["title"]),
                                     str(r.get("genres","")), float(n),
                                     float(r["similarity"]), "Схожість"),
                            unsafe_allow_html=True,
                        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Рекомендації
# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "👤 Рекомендації":
    st.markdown("""
    <div class="page-banner">
      <h1>👤 Персональні рекомендації</h1>
      <p>Отримайте персоналізовані рекомендації для будь-якого користувача</p>
    </div>""", unsafe_allow_html=True)

    if not models_ready() or movies_df is None:
        st.warning("Спочатку навчіть моделі: `python train.py`")
        st.stop()

    left, right = st.columns([3, 7])

    with left:
        st.markdown(f'<div style="font-size:16px;font-weight:700;color:{TEXT_MAIN};margin-bottom:12px">Параметри</div>',
                    unsafe_allow_html=True)
        user_id = st.number_input("User ID", min_value=1, max_value=6040, value=1, step=1)
        method  = st.radio("Метод рекомендацій",
                            ["📝 Content-Based", "🤝 ALS (Collaborative)", "⚡ Гібридний"])
        top_n   = st.slider("Кількість рекомендацій", 5, 20, 10)
        run_btn = st.button("🎯 Отримати рекомендації", use_container_width=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # User profile card
        if ratings_df is not None:
            ur = ratings_df[ratings_df["userId"] == user_id]
            if not ur.empty:
                st.markdown(f"""
                <div class="algo-card" style="padding:16px">
                  <div style="color:{TEXT_MUTED};font-size:12px">Профіль #{user_id}</div>
                  <div style="color:{TEXT_MAIN};font-size:24px;font-weight:800;margin:6px 0">{len(ur):,} оцінок</div>
                  <div style="color:{SUCCESS};font-size:14px">Середня: {ur['rating'].mean():.2f} ⭐</div>
                </div>""", unsafe_allow_html=True)

                # Pie chart of top genres
                ur_movies  = movies_df[movies_df["movieId"].isin(ur["movieId"])]
                genre_cnt  = ur_movies["genres"].str.split("|").explode().value_counts().head(8)
                if not genre_cnt.empty:
                    import plotly.graph_objects as go
                    fig_pie = go.Figure(go.Pie(
                        labels=genre_cnt.index, values=genre_cnt.values, hole=0.4,
                        marker=dict(colors=[
                            "#6C63FF","#FF6B6B","#4ECDC4","#FFE66D",
                            "#A8E6CF","#FF8B94","#B4A7D6","#F9D423"]),
                        textfont=dict(size=11, color=TEXT_MAIN),
                    ))
                    fig_pie.update_layout(
                        paper_bgcolor=BG_DARK, plot_bgcolor=CARD_BG,
                        font=dict(color=TEXT_MAIN), height=220,
                        margin=dict(l=0,r=0,t=10,b=0),
                        legend=dict(bgcolor=CARD_BG, font=dict(size=10)),
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

    with right:
        if run_btn:
            st.session_state["rec_uid"]    = user_id
            st.session_state["rec_method"] = method
            st.session_state["rec_topn"]   = top_n

        uid_s    = st.session_state.get("rec_uid")
        meth_s   = st.session_state.get("rec_method", method)
        topn_s   = st.session_state.get("rec_topn", top_n)

        if uid_s is None:
            st.info("Оберіть параметри та натисніть «Отримати рекомендації».")
        else:
            with st.spinner("Генерація рекомендацій…"):
                recs = pd.DataFrame()
                score_col = "score"
                try:
                    if "Content" in meth_s:
                        recs      = cb_model.recommend_by_user(uid_s, ratings_df, top_n=topn_s)
                        score_col = "cb_score"
                    elif "ALS" in meth_s:
                        recs = cf_model.recommend(uid_s, top_n=topn_s)
                        if recs.empty:
                            st.info(f"Користувач {uid_s} не знайдений у CF-моделі. Використовується CB.")
                            recs = cb_model.recommend_by_user(uid_s, ratings_df, top_n=topn_s)
                            score_col = "cb_score"
                        else:
                            score_col = "cf_score"
                    else:
                        recs = hybrid_model.recommend(uid_s, ratings_df, top_n=topn_s)
                        score_col = "hybrid_score"
                        if not recs.empty and "alpha_used" in recs.columns:
                            alpha_val = float(recs["alpha_used"].iloc[0])
                            activity  = (
                                "холодний старт" if alpha_val >= 0.8
                                else "збалансований" if alpha_val >= 0.4
                                else "активний користувач"
                            )
                            st.markdown(f"""
                            <div style="background:#2A2D3E;border-radius:8px;padding:8px 14px;
                                        display:inline-block;margin-bottom:12px;">
                              <span style="color:{TEXT_MUTED};font-size:12px">Adaptive alpha: </span>
                              <span style="color:{PRIMARY};font-weight:700;font-size:18px">α = {alpha_val:.2f}</span>
                              <span style="color:{TEXT_MUTED};font-size:12px;margin-left:8px">({activity})</span>
                            </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Помилка при генерації рекомендацій: {e}")

            if recs is not None and not recs.empty and score_col in recs.columns:
                sc_raw  = recs[score_col].values.astype(float)
                sc_norm = _norm_series(pd.Series(sc_raw)).values

                pairs = list(zip(recs.itertuples(index=False), sc_norm, sc_raw))
                for row_pair in [pairs[i:i+2] for i in range(0, len(pairs), 2)]:
                    cols = st.columns(2)
                    for col, (row, n, raw) in zip(cols, row_pair):
                        title  = str(row.title)
                        genres = str(getattr(row, "genres", ""))
                        col.markdown(
                            rec_card(title, _year(title), genres, float(n), float(raw),
                                     score_col.replace("_", " ").title()),
                            unsafe_allow_html=True,
                        )

                # Score comparison charts
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                tab_labels = []
                col_map: dict[str, str] = {}
                for lbl, col_name in [("📝 CB", "cb_score"), ("🤝 CF", "cf_score"), ("⚡ Hybrid", "hybrid_score")]:
                    if col_name in recs.columns:
                        tab_labels.append(lbl)
                        col_map[lbl] = col_name

                if tab_labels:
                    import plotly.graph_objects as go
                    short = [t[:28] + "…" if len(t) > 28 else t for t in recs["title"]]
                    for tab, lbl in zip(st.tabs(tab_labels), tab_labels):
                        with tab:
                            sc = recs[col_map[lbl]].values
                            fig = go.Figure(go.Bar(
                                x=sc, y=short, orientation="h",
                                marker=dict(color=sc,
                                            colorscale=[[0, PRIMARY], [1, SECONDARY]]),
                                text=[f"{v:.4f}" for v in sc],
                                textposition="outside",
                                textfont=dict(color=TEXT_MUTED, size=10),
                            ))
                            fig.update_layout(
                                paper_bgcolor=BG_DARK, plot_bgcolor=CARD_BG,
                                font=dict(color=TEXT_MAIN),
                                height=max(250, topn_s * 30),
                                margin=dict(l=0, r=60, t=10, b=20),
                                xaxis=dict(gridcolor="#2A2D3E"),
                                yaxis=dict(gridcolor="#2A2D3E", autorange="reversed"),
                            )
                            st.plotly_chart(fig, use_container_width=True)
            elif uid_s is not None:
                st.info("Рекомендацій не знайдено для цього користувача.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Аналіз та метрики
# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "📊 Аналіз та метрики":
    st.markdown("""
    <div class="page-banner">
      <h1>📊 Аналіз та метрики</h1>
      <p>Оцінка якості алгоритмів рекомендацій</p>
    </div>""", unsafe_allow_html=True)

    if not models_ready():
        st.warning("Спочатку навчіть моделі: `python train.py`")
        st.stop()

    from utils.visualizations import (
        plot_rmse_mae, plot_metrics_comparison,
        plot_tfidf_heatmap, plot_cosine_similarity,
        plot_latent_space, plot_als_training_loss,
        plot_confidence_formula,
    )

    t1, t2, t3, t4, t5 = st.tabs([
        "📉 Якість прогнозування",
        "🎯 Ранжування",
        "🗺️ Матриця взаємодій",
        "🔵 Латентний простір",
        "📐 Confidence formula",
    ])

    # ── Tab 1: RMSE / MAE ─────────────────────────────────────────────────
    with t1:
        ev = (metrics or {}).get("eval_results", {})
        if ev:
            c1, c2 = st.columns([3, 2])
            with c1:
                st.plotly_chart(plot_rmse_mae(ev), use_container_width=True)
            with c2:
                st.markdown(f"""
                <div class="algo-card" style="margin-top:16px">
                  <div class="algo-title">Формули</div>
                  <div class="algo-desc">
                    <b style="color:{PRIMARY}">RMSE</b> = √ mean((y − ŷ)²)<br><br>
                    <b style="color:{SECONDARY}">MAE</b> = mean(|y − ŷ|)<br><br>
                    <b style="color:{SUCCESS}">Baseline</b>: прогнозує глобальну середню оцінку.<br><br>
                    <b style="color:{PRIMARY}">SVD</b>: scipy.sparse.linalg.svds, k=50.
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Метрики не знайдені. Запустіть `python train.py`.")

    # ── Tab 2: Precision / NDCG ───────────────────────────────────────────
    with t2:
        rank_df = (metrics or {}).get("ranking_metrics", pd.DataFrame())
        if rank_df is not None and not rank_df.empty:
            st.plotly_chart(plot_metrics_comparison(rank_df), use_container_width=True)
            st.markdown(f'<div style="font-size:15px;font-weight:600;color:{TEXT_MAIN};margin:12px 0 6px">Зведена таблиця</div>',
                        unsafe_allow_html=True)
            st.dataframe(
                rank_df.style.format({
                    "Precision": "{:.4f}", "Recall": "{:.4f}", "NDCG": "{:.4f}"
                }),
                use_container_width=True,
            )
        else:
            st.info("Метрики ранжування відсутні. Запустіть `python train.py`.")

    # ── Tab 3: TF-IDF heatmaps ────────────────────────────────────────────
    with t3:
        if cb_model and movies_df is not None:
            hc1, hc2 = st.columns(2)
            with hc1:
                with st.spinner("Будуємо TF-IDF матрицю…"):
                    st.plotly_chart(plot_tfidf_heatmap(cb_model, movies_df), use_container_width=True)
            with hc2:
                with st.spinner("Будуємо матрицю схожості…"):
                    st.plotly_chart(plot_cosine_similarity(cb_model, movies_df), use_container_width=True)
        else:
            st.info("Завантажте моделі.")

    # ── Tab 4: Latent space ───────────────────────────────────────────────
    with t4:
        if cf_model and movies_df is not None:
            lc1, lc2 = st.columns(2)
            with lc1:
                with st.spinner("PCA…"):
                    st.plotly_chart(plot_latent_space(cf_model, movies_df), use_container_width=True)
            with lc2:
                st.plotly_chart(plot_als_training_loss(cf_model), use_container_width=True)
        else:
            st.info("Завантажте моделі.")

    # ── Tab 5: Confidence formula ─────────────────────────────────────────
    with t5:
        cc1, cc2 = st.columns([3, 2])
        with cc1:
            st.plotly_chart(plot_confidence_formula(), use_container_width=True)
        with cc2:
            st.markdown(f"""
            <div class="algo-card">
              <div class="algo-title">Формула confidence ALS</div>
              <div class="algo-desc">
                <b style="color:{PRIMARY}">c(u,i) = 1 + α · r(u,i)</b><br><br>
                де <b>r(u,i)</b> — оцінка, <b>α</b> — параметр масштабування (α=40).<br><br>
                Навіть незафіксовані взаємодії (r=0) мають confidence=1,
                а оцінки > 0 суттєво підвищують довіру моделі до спостереження.<br><br>
                ALS мінімізує:<br>
                Σ c(u,i)·(pref − p·q)² + λ(‖P‖²+‖Q‖²)
              </div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Про систему
# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "ℹ️ Про систему":
    st.markdown("""
    <div class="page-banner">
      <h1>ℹ️ Про систему</h1>
      <p>Архітектура, технологічний стек та алгоритми</p>
    </div>""", unsafe_allow_html=True)

    st.markdown(f'<div style="font-size:19px;font-weight:700;color:{TEXT_MAIN};margin-bottom:14px">Архітектура</div>',
                unsafe_allow_html=True)
    ac1, ac2, ac3 = st.columns(3)
    for col, icon, title, desc in [
        (ac1, "📦", "Шар даних",
         "• DataLoader — MovieLens 1M<br>"
         "• movies / ratings / users<br>"
         "• Попередня обробка: year, genres_str, avg_rating<br>"
         "• Encoding latin-1, sep='::'"),
        (ac2, "⚙️", "Шар моделей",
         "• ContentBasedRecommender<br>"
         "• CollaborativeFilterRecommender (ALS)<br>"
         "• HybridRecommender<br>"
         "• MetricsEvaluator (SVD, RMSE, NDCG)"),
        (ac3, "🖥️", "Шар інтерфейсу",
         "• Streamlit — веб-застосунок<br>"
         "• Plotly — інтерактивні графіки<br>"
         "• Custom CSS — темна тема<br>"
         "• joblib — серіалізація моделей"),
    ]:
        col.markdown(f"""
        <div class="algo-card">
          <div class="algo-icon">{icon}</div>
          <div class="algo-title">{title}</div>
          <div class="algo-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:19px;font-weight:700;color:{TEXT_MAIN};margin-bottom:12px">Стек технологій</div>',
                unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Технологія": ["Python 3.13","Streamlit","scikit-learn","scipy","Plotly","pandas","numpy","joblib"],
        "Версія":     ["3.13.4","≥1.40","≥1.5","≥1.14","≥5.24","≥2.2","≥2.0","≥1.4"],
        "Призначення":["Мова програмування","Веб-інтерфейс","TF-IDF / NearestNeighbors / PCA",
                       "SVD / лінійна алгебра","Інтерактивні графіки","Обробка даних",
                       "Числові обчислення","Серіалізація моделей"],
    }), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:19px;font-weight:700;color:{TEXT_MAIN};margin-bottom:12px">Порівняння алгоритмів</div>',
                unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Алгоритм":    ["Content-Based","Collaborative (ALS)","Hybrid"],
        "Підхід":      ["TF-IDF + Cosine Similarity","Matrix Factorization (ALS)","Weighted Combination"],
        "Переваги":    ["Не потребує оцінок інших","Виявляє латентні уподобання","Вирішує холодний старт"],
        "Недоліки":    ["Обмежений жанром","Холодний старт нових","Складніший у навчанні"],
        "Складність":  ["O(n·m)","O(k·n·m)","O(k·n·m)"],
    }), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    ds1, ds2 = st.columns(2)
    with ds1:
        st.markdown(f"""
        <div class="algo-card">
          <div class="algo-title">MovieLens 1M</div>
          <div class="algo-desc">
            • 1,000,209 анонімних оцінок<br>
            • 3,883 фільми (1995–2000)<br>
            • 6,040 користувачів<br>
            • Шкала оцінок: 1–5<br>
            • Кожен користувач оцінив ≥20 фільмів
          </div>
        </div>""", unsafe_allow_html=True)
    with ds2:
        st.markdown(f"""
        <div class="algo-card">
          <div class="algo-title">Цитування</div>
          <div class="algo-desc">
            F. Maxwell Harper and Joseph A. Konstan (2015).<br>
            <i>The MovieLens Datasets: History and Context.</i><br>
            ACM TiiS 5, 4, Article 19.<br><br>
            <a href="https://grouplens.org/datasets/movielens/1m/"
               style="color:{PRIMARY}">grouplens.org/datasets/movielens/1m/</a>
          </div>
        </div>""", unsafe_allow_html=True)
