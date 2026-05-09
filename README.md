# Film Recommendation System

Система рекомендацій фільмів — курсова робота з «Алгоритмізації та програмування» (спеціалізація Data Science).

## Опис

Веб-застосунок на базі **Streamlit** з трьома алгоритмами рекомендацій:

- **Content-Based Filtering** — TF-IDF векторизація жанрів + косинусна схожість
- **Collaborative Filtering (ALS)** — матрична факторизація з неявним зворотним зв'язком
- **Hybrid** — адаптивна зважена комбінація CB та CF

Датасет: **MovieLens 1M** (1,000,209 оцінок, 3,883 фільми, 6,040 користувачів).

---

## Встановлення

```bash
pip install -r requirements.txt
```

## Підготовка моделей

```bash
python train.py
```

> Час навчання: ~3–5 хвилин на CPU. Моделі зберігаються у папці `models/`.

## Запуск

```bash
streamlit run app.py
```

Відкрийте `http://localhost:8501`.

---

## Структура проекту

```
film_recommender/
├── app.py                    — Streamlit застосунок (5 сторінок)
├── train.py                  — Скрипт навчання моделей
├── recommenders/
│   ├── content_based.py      — ContentBasedRecommender
│   ├── collaborative.py      — CollaborativeFilterRecommender (ALS)
│   ├── hybrid.py             — HybridRecommender
│   └── evaluator.py          — MetricsEvaluator
├── utils/
│   ├── data_loader.py        — DataLoader
│   └── visualizations.py     — Plotly графіки
├── data/ml-1m/               — MovieLens 1M датасет
└── models/                   — Збережені моделі (.pkl)
```

---

## Алгоритми

### Content-Based Filtering

```
TF-IDF: W(i,j) = TF(i,j) × log(N / DF(i))

Cosine similarity: cos(q, d) = (q · d) / (||q|| × ||d||)
```

Для кожного користувача будується профіль жанрів — зважена сума TF-IDF векторів оцінених фільмів. Рекомендуються найближчі фільми у жанровому просторі.

### Collaborative Filtering (ALS)

```
Confidence:  c(u,i) = 1 + α × r(u,i),  α = 40

Objective: min Σ c(u,i)(p(u,i) - P[u]·Q[i])² + λ(||P||² + ||Q||²)

ALS update:
  P[u] = (Q^T C^u Q + λI)^-1 Q^T C^u p(u)
  Q[i] = (P^T C^i P + λI)^-1 P^T C^i p(i)
```

Параметри: `factors=100`, `iterations=20`, `regularization=0.01`.

### Hybrid Recommender

```
score = α × cb_score + (1 - α) × cf_score

Adaptive α:
  n < 10    → α = 0.80  (cold start)
  10–50     → α = 0.60
  50–200    → α = 0.40
  n > 200   → α = 0.20  (active user)
```

### Метрики оцінки

```
RMSE = √( mean( (y_true - y_pred)² ) )
MAE  = mean( |y_true - y_pred| )

Precision@K = |relevant ∩ recommended| / K
Recall@K    = |relevant ∩ recommended| / |relevant|
NDCG@K = DCG@K / IDCG@K,  DCG = Σ rel_i / log₂(i+2)
```

---

## Датасет

F. Maxwell Harper and Joseph A. Konstan (2015). *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19.

[https://grouplens.org/datasets/movielens/1m/](https://grouplens.org/datasets/movielens/1m/)
