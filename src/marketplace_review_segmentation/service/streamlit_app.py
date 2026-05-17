from __future__ import annotations

from pathlib import Path
import sys

import duckdb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from marketplace_review_segmentation.config import settings

# ── Palette (matches presentation) ───────────────────────────────────────────
ORANGE     = "#F5A623"
GREEN      = "#9DC54A"
PURPLE     = "#B87ACC"
NAVY       = "#1E2270"
LIGHT_BLUE = "#A8BCE8"
GRAY_TEXT  = "#666666"
DARK_TEXT  = "#1C1C1C"

SEG_COLORS = {0: GREEN, 1: LIGHT_BLUE, 2: ORANGE, 3: PURPLE, 4: "#E07878"}
SEG_NAMES  = {
    0: "Лояльные и активные",
    1: "Максимально позитивные",
    2: "Сверхактивные мультибрендовые",
    3: "Критичные активные",
    4: "Умеренно проблемные",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reviewer Segmentation",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

/* push tabs below hero */
.block-container {{ padding-top: 1.5rem; }}
[data-testid="stTabs"] {{ margin-top: 1.5rem; }}

/* ── Hero banner ── */
.hero {{
    background: linear-gradient(135deg, {NAVY} 0%, #2d3596 100%);
    border-radius: 16px;
    padding: 2.2rem 2.5rem;
    color: white;
    margin-bottom: 0.5rem;
}}
.hero h1 {{
    font-size: 1.65rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    color: white;
    line-height: 1.3;
}}
.hero p {{
    font-size: 0.95rem;
    opacity: 0.75;
    margin: 0;
}}

/* ── Metric card ── */
.metric-card {{
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    border: 1.5px solid #EEEEEE;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    text-align: center;
}}
.metric-card .value {{
    font-size: 2rem;
    font-weight: 700;
    color: {NAVY};
    line-height: 1.1;
}}
.metric-card .label {{
    font-size: 0.82rem;
    color: {GRAY_TEXT};
    margin-top: 0.3rem;
    font-weight: 500;
}}

/* ── Badge / chip ── */
.badge {{
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: white;
}}

/* ── Section header ── */
.section-header {{
    font-size: 1.1rem;
    font-weight: 700;
    color: {DARK_TEXT};
    border-left: 4px solid {ORANGE};
    padding-left: 0.75rem;
    margin: 1.5rem 0 0.8rem 0;
}}

/* ── Segment card ── */
.seg-card {{
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    border: 1.5px solid #EEEEEE;
    background: white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    margin-bottom: 0.5rem;
}}
.seg-card .seg-title {{
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}}
.seg-card .seg-stat {{
    font-size: 0.85rem;
    color: {GRAY_TEXT};
    line-height: 1.7;
}}

/* ── Pipeline step ── */
.pipe-step {{
    background: white;
    border-radius: 10px;
    border: 1.5px solid #EEEEEE;
    padding: 0.9rem 1rem;
    text-align: center;
    font-size: 0.85rem;
    font-weight: 600;
    color: {DARK_TEXT};
}}
.pipe-step .pipe-sub {{ font-weight: 400; color: {GRAY_TEXT}; font-size: 0.78rem; }}

/* ── Tech badge ── */
.tech-badge {{
    display: inline-block;
    background: #F0F0F8;
    color: {NAVY};
    border-radius: 6px;
    padding: 0.25rem 0.65rem;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 0.2rem 0.15rem;
    border: 1px solid #DDDDF0;
}}

/* ── API block ── */
.api-endpoint {{
    background: #F8F9FD;
    border-radius: 10px;
    border: 1.5px solid #E0E4F0;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}}
.api-method {{
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 5px;
    font-size: 0.8rem;
    font-weight: 700;
    color: white;
    margin-right: 0.6rem;
}}
.method-get  {{ background: {GREEN}; }}
.method-post {{ background: {ORANGE}; }}

/* ── Info block ── */
.info-box {{
    background: #F5F5FF;
    border-left: 4px solid {PURPLE};
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: {DARK_TEXT};
    margin: 0.8rem 0;
}}

/* model comparison table */
.model-table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
.model-table th {{ background: {NAVY}; color: white; padding: 0.6rem 1rem; text-align: left; }}
.model-table td {{ padding: 0.55rem 1rem; border-bottom: 1px solid #EEEEEE; }}
.model-table tr:last-child td {{ border-bottom: none; }}
.model-table tr.best-row td {{ background: #F0FBE8; font-weight: 600; }}
.highlight {{ color: {GREEN}; font-weight: 700; }}
</style>
""", unsafe_allow_html=True)

# ── DB connection ─────────────────────────────────────────────────────────────
_DEMO_PATH = ROOT / "data" / "artifacts" / "demo.duckdb"

@st.cache_resource
def get_conn() -> tuple[duckdb.DuckDBPyConnection, bool]:
    try:
        conn = duckdb.connect(str(settings.duckdb_path), read_only=True)
        conn.execute("SELECT 1 FROM gold_author_segments LIMIT 1")
        return conn, False
    except Exception:
        pass
    if _DEMO_PATH.exists():
        return duckdb.connect(str(_DEMO_PATH), read_only=True), True
    return None, True  # type: ignore[return-value]

def safe_query(query: str) -> pd.DataFrame | None:
    conn, _ = get_conn()
    if conn is None:
        return None
    try:
        return conn.execute(query).fetchdf()
    except Exception:
        return None

def _is_demo() -> bool:
    _, demo = get_conn()
    return demo

# ── Demo mode banner ──────────────────────────────────────────────────────────
if _is_demo():
    st.info(
        "**Демо-режим** — данные из встроенной выборки (demo.duckdb). "
        "Для полного набора запустите локально с реальным хранилищем.",
        icon="ℹ️",
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Обзор",
    "Данные",
    "Модели",
    "Сегменты",
    "API",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ОБЗОР
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="hero">
        <h1>Анализ и сегментация авторов отзывов на товары маркетплейсов<br>на основе их поведенческих паттернов</h1>
        <p>Выявление поведенческих паттернов авторов на основе многодоменных признаков · HSE · 2026</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in [
        (c1, "91 335",  "авторов в анализе"),
        (c2, "950 518", "отзывов в датасете"),
        (c3, "5",       "сегментов"),
        (c4, "32",      "поведенческих признаков"),
        (c5, "3 мес.",  "период наблюдения"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{val}</div>
                <div class="label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="section-header">О проекте</div>', unsafe_allow_html=True)
        st.markdown("""
        Авторы отзывов на маркетплейсах — неоднородная аудитория с разными поведенческими профилями.
        Цель работы — **сегментировать авторов** на основе многодоменных поведенческих признаков
        без заранее заданной разметки (обучение без учителя).

        Объект анализа: пользователи, написавшие **≥ 3 отзывов** за период наблюдения.
        Признаки охватывают четыре домена: отзывная активность, покупки, чеки и действия на маркетплейсе.
        """)

        st.markdown('<div class="section-header">Датасет</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
            <b>T-ECD</b> — кросс-доменный датасет от T-Tech (T-Банк), Hugging Face<br>
            <span style="color:{GRAY_TEXT}">t-tech/T-ECD · CC-BY-NC-SA-4.0</span><br><br>
            Включает: отзывы, транзакции, чеки, события маркетплейса, справочники брендов и товаров.
            Сырые тексты отзывов не публикуются — вместо них <b>предобученные эмбеддинги (312 измерений)</b>.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Технологический стек</div>', unsafe_allow_html=True)
        techs = ["DuckDB", "Apache Airflow", "FastAPI", "Streamlit",
                 "scikit-learn", "KMeans", "Python 3.11", "Plotly"]
        badges = "".join(f'<span class="tech-badge">{t}</span>' for t in techs)
        st.markdown(badges, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-header">Архитектура пайплайна</div>', unsafe_allow_html=True)

        steps = [
            ("Bronze",   "Сырые parquet\nT-ECD → DuckDB"),
            ("Silver",   "Нормализация\nдоменов и событий"),
            ("Gold",     "Аналитические\nвитрины и DM"),
            ("Features", "32 поведенческих\nпризнака / автор"),
            ("Model",    "KMeans k=5\nSilhouette=0.190"),
            ("Service",  "FastAPI + Streamlit\nдашборд"),
        ]
        cols = st.columns(3)
        for i, (title, sub) in enumerate(steps):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="pipe-step">
                    <b>{title}</b><br>
                    <span class="pipe-sub">{sub}</span>
                </div>""", unsafe_allow_html=True)
            if i == 2:
                st.markdown("")

        st.markdown('<div class="section-header">Результаты сегментации</div>',
                    unsafe_allow_html=True)
        for sid, name in SEG_NAMES.items():
            color = SEG_COLORS[sid]
            st.markdown(
                f'<span class="badge" style="background:{color}; margin:3px">'
                f'Сегмент {sid} · {name}</span>',
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ДАННЫЕ
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:

    # ── Блок 1: Отзывы ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Распределение отзывов</div>',
                unsafe_allow_html=True)

    if _is_demo():
        rating_data = safe_query(
            "SELECT CAST(rating AS INTEGER) AS rating, cnt FROM fact_reviews "
            "WHERE rating BETWEEN 1 AND 5 ORDER BY 1"
        )
    else:
        rating_data = safe_query(
            "SELECT CAST(rating AS INTEGER) AS rating, COUNT(*) AS cnt "
            "FROM fact_reviews WHERE rating BETWEEN 1 AND 5 GROUP BY 1 ORDER BY 1"
        )

    if _is_demo():
        reviews_dist = safe_query(
            "SELECT reviews_count, authors FROM dm_author_review_dist ORDER BY reviews_count LIMIT 20"
        )
    else:
        reviews_dist = safe_query(
            "SELECT reviews_count, COUNT(*) AS authors FROM dm_author_features "
            "GROUP BY reviews_count ORDER BY reviews_count LIMIT 20"
        )

    col_r, col_u = st.columns(2)

    with col_r:
        if rating_data is not None and not rating_data.empty:
            total_r = rating_data["cnt"].sum()
            rating_data["share"] = rating_data["cnt"] / total_r * 100
            rating_data["label"] = rating_data.apply(
                lambda r: f"{r['share']:.1f}%<br>({int(r['cnt']):,})", axis=1
            )
            fig = go.Figure(go.Bar(
                x=rating_data["rating"],
                y=rating_data["cnt"],
                marker_color=["#CCCCCC", "#CCCCCC", "#CCCCCC", "#CCCCCC", ORANGE],
                text=rating_data["label"],
                textposition="outside",
            ))
            fig.update_layout(
                title="Оценки отзывов",
                xaxis=dict(tickvals=[1,2,3,4,5], ticktext=["1★","2★","3★","4★","5★"], title="Оценка"),
                yaxis=dict(title="Число отзывов", tickformat=".0s"),
                plot_bgcolor="white", paper_bgcolor="white",
                showlegend=False, height=360,
                margin=dict(t=50, b=40),
                font=dict(family="Inter"),
            )
            fig.update_yaxes(showgrid=True, gridcolor="#F0F0F0")
            st.plotly_chart(fig, use_container_width=True)

    with col_u:
        if reviews_dist is not None and not reviews_dist.empty:
            fig2 = go.Figure(go.Bar(
                x=reviews_dist["reviews_count"].astype(str),
                y=reviews_dist["authors"],
                marker_color=LIGHT_BLUE,
            ))
            fig2.update_layout(
                title="Число отзывов на автора (топ-20 значений)",
                xaxis_title="Отзывов на автора",
                yaxis_title="Авторов",
                plot_bgcolor="white", paper_bgcolor="white",
                height=360, margin=dict(t=50, b=40),
                font=dict(family="Inter"),
            )
            fig2.update_yaxes(showgrid=True, gridcolor="#F0F0F0")
            st.plotly_chart(fig2, use_container_width=True)

    # ── Блок 2: Временная динамика ────────────────────────────────────────────
    st.markdown('<div class="section-header">Временная динамика</div>', unsafe_allow_html=True)

    ratings_dyn   = safe_query("SELECT * FROM gold_rating_dynamics ORDER BY review_date")
    tx_dyn        = safe_query("SELECT * FROM gold_transaction_dynamics ORDER BY payment_date")
    mp_data       = safe_query("SELECT * FROM gold_marketplace_dynamics ORDER BY event_date")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if ratings_dyn is not None and not ratings_dyn.empty:
            fig3 = px.line(
                ratings_dyn, x="review_date", y="avg_rating",
                title="Средняя оценка по дням",
                color_discrete_sequence=[NAVY],
            )
            fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               height=300, font=dict(family="Inter"),
                               margin=dict(t=50, b=40))
            fig3.update_yaxes(showgrid=True, gridcolor="#F0F0F0")
            st.plotly_chart(fig3, use_container_width=True)

    with col_d2:
        if ratings_dyn is not None and not ratings_dyn.empty and "reviews_count" in ratings_dyn.columns:
            fig_rc = px.bar(
                ratings_dyn, x="review_date", y="reviews_count",
                title="Число отзывов по дням",
                color_discrete_sequence=[LIGHT_BLUE],
            )
            fig_rc.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                 height=300, font=dict(family="Inter"),
                                 margin=dict(t=50, b=40))
            fig_rc.update_yaxes(showgrid=True, gridcolor="#F0F0F0")
            st.plotly_chart(fig_rc, use_container_width=True)

    col_d3, col_d4 = st.columns(2)
    with col_d3:
        if tx_dyn is not None and not tx_dyn.empty:
            fig_tx = px.line(
                tx_dyn, x="payment_date", y="transactions_count",
                title="Транзакции по дням",
                color_discrete_sequence=[ORANGE],
            )
            fig_tx.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                 height=300, font=dict(family="Inter"),
                                 margin=dict(t=50, b=40))
            fig_tx.update_yaxes(showgrid=True, gridcolor="#F0F0F0")
            st.plotly_chart(fig_tx, use_container_width=True)

    with col_d4:
        if mp_data is not None and not mp_data.empty:
            fig5 = px.area(
                mp_data, x="event_date", y="marketplace_events_count",
                title="Число событий на маркетплейсе по дням",
                color_discrete_sequence=[LIGHT_BLUE],
            )
            fig5.update_traces(line_color=NAVY, fillcolor="rgba(168,188,232,0.25)")
            fig5.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               height=300, font=dict(family="Inter"),
                               margin=dict(t=50, b=40))
            fig5.update_yaxes(showgrid=True, gridcolor="#F0F0F0")
            st.plotly_chart(fig5, use_container_width=True)

    # ── Блок 3: Бренды ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Бренды</div>', unsafe_allow_html=True)

    brands_data = safe_query(
        "SELECT * FROM gold_brand_review_summary ORDER BY reviews_count DESC LIMIT 15"
    )

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if brands_data is not None and not brands_data.empty:
            fig4 = px.bar(
                brands_data, x="brand_label", y="reviews_count",
                title="Топ-15 брендов по числу отзывов",
                color_discrete_sequence=[PURPLE],
                hover_data=["avg_rating", "authors_count"] if "authors_count" in brands_data.columns else ["avg_rating"],
            )
            fig4.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               height=360, font=dict(family="Inter"),
                               margin=dict(t=50, b=60),
                               xaxis_tickangle=-35)
            fig4.update_yaxes(showgrid=True, gridcolor="#F0F0F0")
            st.plotly_chart(fig4, use_container_width=True)

    with col_b2:
        if brands_data is not None and not brands_data.empty and "avg_rating" in brands_data.columns:
            fig_br = px.bar(
                brands_data.sort_values("avg_rating"),
                x="avg_rating", y="brand_label",
                orientation="h",
                title="Средняя оценка топ-15 брендов",
                color="avg_rating",
                color_continuous_scale=[[0, "#E07878"], [0.5, ORANGE], [1, GREEN]],
                range_color=[1, 5],
            )
            fig_br.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                 height=360, font=dict(family="Inter"),
                                 margin=dict(t=50, b=40),
                                 showlegend=False,
                                 coloraxis_showscale=False)
            fig_br.update_xaxes(range=[0, 5.2], showgrid=True, gridcolor="#F0F0F0")
            st.plotly_chart(fig_br, use_container_width=True)

    # ── Блок 4: Транзакции ───────────────────────────────────────────────────
    if tx_dyn is not None and not tx_dyn.empty:
        st.markdown('<div class="section-header">Транзакции</div>', unsafe_allow_html=True)

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            fig_amt = px.line(
                tx_dyn, x="payment_date", y="avg_transaction_amount",
                title="Средний чек по дням",
                color_discrete_sequence=[GREEN],
            )
            fig_amt.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                  height=300, font=dict(family="Inter"),
                                  margin=dict(t=50, b=40))
            fig_amt.update_yaxes(showgrid=True, gridcolor="#F0F0F0")
            st.plotly_chart(fig_amt, use_container_width=True)

        with col_t2:
            fig_au = px.line(
                tx_dyn, x="payment_date", y="active_users_count",
                title="Активные пользователи (по транзакциям)",
                color_discrete_sequence=[PURPLE],
            )
            fig_au.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                 height=300, font=dict(family="Inter"),
                                 margin=dict(t=50, b=40))
            fig_au.update_yaxes(showgrid=True, gridcolor="#F0F0F0")
            st.plotly_chart(fig_au, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — МОДЕЛИ
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Сравнение моделей кластеризации</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <table class="model-table">
      <thead>
        <tr>
          <th>Модель</th><th>k</th><th>Silhouette ↑</th>
          <th>Davies-Bouldin ↓</th><th>Примечание</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>KMeans</td><td>4</td><td>0.197</td><td>1.979</td>
            <td>Максимум Silhouette</td></tr>
        <tr class="best-row">
          <td><b>KMeans</b></td><td><b>5</b></td>
          <td><span class="highlight">0.190</span></td>
          <td><span class="highlight">1.868</span></td>
          <td>Выбранная модель</td>
        </tr>
        <tr><td>Gaussian Mixture</td><td>5</td><td>0.070</td><td>3.607</td>
            <td>Слабое разбиение</td></tr>
        <tr><td>Agglomerative</td><td>5</td><td>0.117</td><td>2.414</td>
            <td>Вычислительно дорого</td></tr>
        <tr><td>KMeans + эмбеддинги</td><td>5</td><td>0.014</td><td>4.171</td>
            <td>Исследовательский контур</td></tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)

    # k search chart
    st.markdown('<div class="section-header">Поиск оптимального числа кластеров</div>',
                unsafe_allow_html=True)

    k_vals = [2, 3, 4, 5, 6, 7, 8]
    sil    = [0.1837, 0.1960, 0.1974, 0.1900, 0.1382, 0.1409, 0.1009]
    db_idx = [2.312, 2.089, 1.979, 1.868, 2.643, 2.511, 3.124]

    fig_k = make_subplots(rows=1, cols=2,
                          subplot_titles=("Silhouette Score (выше лучше)",
                                          "Davies-Bouldin Index (ниже лучше)"))
    fig_k.add_trace(go.Scatter(
        x=k_vals, y=sil, mode="lines+markers",
        marker=dict(size=9, color=[GREEN if k == 5 else NAVY for k in k_vals],
                    line=dict(width=2, color="white")),
        line=dict(color=NAVY, width=2.5), name="Silhouette",
    ), row=1, col=1)
    fig_k.add_vline(x=5, line_dash="dash", line_color=GREEN,
                    annotation_text="k=5", row=1, col=1)

    fig_k.add_trace(go.Scatter(
        x=k_vals, y=db_idx, mode="lines+markers",
        marker=dict(size=9, color=[GREEN if k == 5 else PURPLE for k in k_vals],
                    line=dict(width=2, color="white")),
        line=dict(color=PURPLE, width=2.5), name="Davies-Bouldin",
    ), row=1, col=2)
    fig_k.add_vline(x=5, line_dash="dash", line_color=GREEN,
                    annotation_text="k=5", row=1, col=2)

    fig_k.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        height=380, showlegend=False,
        font=dict(family="Inter", size=12),
        margin=dict(t=60, b=40),
    )
    fig_k.update_xaxes(tickmode="linear", dtick=1, title_text="k", showgrid=False)
    fig_k.update_yaxes(showgrid=True, gridcolor="#F0F0F0")
    st.plotly_chart(fig_k, use_container_width=True)

    # Features
    st.markdown('<div class="section-header">Пространство признаков</div>',
                unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    feat_groups = [
        ("Отзывная активность",
         ["reviews_count", "avg_rating", "rating_stddev",
          "share_positive", "share_negative", "avg_review_length",
          "active_days", "brand_diversity", "category_diversity", "recency_days"]),
        ("Транзакции",
         ["transactions_count", "payment_events_count", "avg_transaction_amount",
          "total_transaction_amount", "payment_brand_diversity",
          "transaction_recency_days", "transaction_days"]),
        ("Чеки",
         ["receipts_count", "total_items_purchased", "avg_items_per_receipt",
          "avg_receipt_amount", "total_receipt_amount", "purchased_item_diversity"]),
        ("Маркетплейс",
         ["marketplace_events_count", "viewed_item_diversity",
          "marketplace_subdomain_diversity", "os_diversity",
          "share_view_actions", "marketplace_active_days",
          "marketplace_recency_days", "review_to_receipt_ratio",
          "event_to_receipt_ratio"]),
    ]
    for col, (title, feats) in zip([f1, f2, f3, f4], feat_groups):
        with col:
            st.markdown(f"**{title}**")
            for f in feats:
                st.markdown(f"<small style='color:{GRAY_TEXT}'>· {f}</small>",
                            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — СЕГМЕНТЫ
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    seg_profiles_data = [
        dict(id=0, name="Лояльные и активные",          n=46982, reviews=5.94,
             rating=4.86, neg=0.2,  brands=5.55, color=GREEN),
        dict(id=1, name="Максимально позитивные",        n=17611, reviews=4.98,
             rating=4.89, neg=0.07, brands=4.43, color=LIGHT_BLUE),
        dict(id=2, name="Сверхактивные мультибрендовые", n=9928,  reviews=20.7,
             rating=4.71, neg=7.0,  brands=17.5, color=ORANGE),
        dict(id=3, name="Критичные активные",            n=11391, reviews=8.75,
             rating=4.02, neg=15.4, brands=7.97, color=PURPLE),
        dict(id=4, name="Умеренно проблемные",           n=5423,  reviews=5.58,
             rating=4.04, neg=11.4, brands=5.12, color="#E07878"),
    ]

    # Segment cards
    st.markdown('<div class="section-header">Профили сегментов</div>',
                unsafe_allow_html=True)
    cols5 = st.columns(5)
    for seg, col in zip(seg_profiles_data, cols5):
        with col:
            st.markdown(f"""
            <div class="seg-card">
                <div class="seg-title" style="color:{seg['color']}">
                    Сегмент {seg['id']}
                </div>
                <div style="font-size:0.78rem; color:{GRAY_TEXT}; font-weight:600;
                            margin-bottom:0.5rem">{seg['name']}</div>
                <div class="seg-stat">
                    <b>{seg['n']:,}</b> авторов<br>
                    {seg['reviews']} отзывов (avg)<br>
                    рейтинг {seg['rating']}<br>
                    негатив {seg['neg']}%<br>
                    брендов {seg['brands']}
                </div>
            </div>""", unsafe_allow_html=True)

    # Comparison chart
    st.markdown('<div class="section-header">Сравнение сегментов</div>',
                unsafe_allow_html=True)

    df_seg = pd.DataFrame(seg_profiles_data)
    metric_col = st.selectbox(
        "Метрика для сравнения",
        options=["reviews", "rating", "neg", "brands", "n"],
        format_func=lambda x: {
            "reviews": "Среднее число отзывов",
            "rating":  "Средний рейтинг",
            "neg":     "Доля негативных отзывов (%)",
            "brands":  "Разнообразие брендов",
            "n":       "Число авторов",
        }[x],
    )

    fig_seg = go.Figure(go.Bar(
        x=[f"Сег. {r['id']}" for _, r in df_seg.iterrows()],
        y=df_seg[metric_col],
        marker_color=[r["color"] for _, r in df_seg.iterrows()],
        text=df_seg[metric_col],
        textposition="outside",
    ))
    fig_seg.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        height=360, showlegend=False,
        font=dict(family="Inter", size=13),
        margin=dict(t=30, b=40),
    )
    fig_seg.update_yaxes(showgrid=True, gridcolor="#F0F0F0")
    st.plotly_chart(fig_seg, use_container_width=True)

    # Author explorer
    st.markdown('<div class="section-header">Просмотр авторов по сегменту</div>',
                unsafe_allow_html=True)
    authors_df = safe_query(
        "SELECT * FROM gold_author_segments ORDER BY reviews_count DESC"
    )
    if authors_df is not None and not authors_df.empty:
        available = sorted(authors_df["final_segment_id"].dropna().unique().tolist())
        sel = st.selectbox(
            "Выберите сегмент",
            available,
            format_func=lambda x: f"Сегмент {int(x)} — {SEG_NAMES.get(int(x), '')}",
        )
        filtered = authors_df[authors_df["final_segment_id"] == sel].head(200)

        # Drop columns that are entirely null or entirely zero
        non_empty = [
            c for c in filtered.columns
            if filtered[c].notna().any() and not (filtered[c].fillna(0) == 0).all()
        ]
        st.caption(f"Авторов в сегменте: {len(authors_df[authors_df['final_segment_id'] == sel]):,}")
        st.dataframe(filtered[non_empty], use_container_width=True, hide_index=True)
    else:
        st.info("Сегменты ещё не рассчитаны. Запустите этап обучения модели.")

    # Transaction profile
    seg_tx = safe_query(
        "SELECT * FROM gold_segment_transaction_summary ORDER BY final_segment_id"
    )
    if seg_tx is not None and not seg_tx.empty:
        st.markdown('<div class="section-header">Транзакционный профиль сегментов</div>',
                    unsafe_allow_html=True)
        cols_to_plot = [c for c in ["avg_total_transaction_amount", "avg_transactions_count"]
                        if c in seg_tx.columns]
        if cols_to_plot:
            fig_tx = px.bar(
                seg_tx,
                x="final_segment_id",
                y=cols_to_plot,
                barmode="group",
                title="Транзакционный профиль сегментов",
                color_discrete_sequence=[NAVY, LIGHT_BLUE],
                labels={"final_segment_id": "Сегмент", "value": "Значение"},
            )
            fig_tx.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                 height=340, font=dict(family="Inter"),
                                 margin=dict(t=50, b=40))
            fig_tx.update_yaxes(showgrid=True, gridcolor="#F0F0F0")
            st.plotly_chart(fig_tx, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — API
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f"""
    <div class="hero">
        <h1>REST API — Reviewer Segmentation</h1>
        <p>FastAPI · версия 1.0.0 · локально: http://localhost:8000</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Эндпоинты</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="api-endpoint">
        <span class="api-method method-get">GET</span>
        <code style="font-size:1rem">/health</code>
        <p style="margin:0.6rem 0 0.3rem; color:#444; font-size:0.9rem">
            Проверка доступности сервиса
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Пример ответа /health"):
        st.code('{"status": "ok"}', language="json")

    st.markdown("""
    <div class="api-endpoint">
        <span class="api-method method-get">GET</span>
        <code style="font-size:1rem">/segment/{user_id}</code>
        <p style="margin:0.6rem 0 0.3rem; color:#444; font-size:0.9rem">
            Получить сегмент автора по <code>user_id</code>.
            Возвращает baseline, NLP и итоговый сегмент.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_req, col_resp = st.columns(2)
    with col_req:
        st.markdown("**Запрос**")
        st.code("curl http://localhost:8000/segment/USER_123", language="bash")
    with col_resp:
        st.markdown("**Ответ**")
        st.code("""{
  "user_id": "USER_123",
  "baseline_segment_id": 2,
  "nlp_segment_id": 2,
  "final_segment_id": 2
}""", language="json")

    st.markdown("""
    <div class="api-endpoint">
        <span class="api-method method-get">GET</span>
        <code style="font-size:1rem">/docs</code>
        <p style="margin:0.6rem 0 0.3rem; color:#444; font-size:0.9rem">
            Swagger UI — интерактивная документация API (автогенерация FastAPI)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Live tester
    st.markdown('<div class="section-header">Интерактивный поиск</div>',
                unsafe_allow_html=True)
    st.markdown("Введите `user_id` для получения сегмента напрямую из базы:")

    uid_input = st.text_input("user_id", placeholder="например: 12345")
    if uid_input:
        row = safe_query(f"""
            SELECT user_id, baseline_segment_id, nlp_segment_id, final_segment_id
            FROM gold_author_segments
            WHERE CAST(user_id AS VARCHAR) = '{uid_input}'
            LIMIT 1
        """)
        if row is not None and not row.empty:
            r = row.iloc[0]
            sid = int(r["final_segment_id"]) if pd.notna(r["final_segment_id"]) else None
            color = SEG_COLORS.get(sid, "#999") if sid is not None else "#999"
            name  = SEG_NAMES.get(sid, "—") if sid is not None else "—"
            c_a, c_b, c_c = st.columns(3)
            c_a.metric("Baseline сегмент", int(r["baseline_segment_id"])
                       if pd.notna(r["baseline_segment_id"]) else "—")
            c_b.metric("NLP сегмент", int(r["nlp_segment_id"])
                       if pd.notna(r["nlp_segment_id"]) else "—")
            c_c.metric("Итоговый сегмент", sid if sid is not None else "—")
            st.markdown(
                f'<span class="badge" style="background:{color}; font-size:1rem; '
                f'padding:0.4rem 1rem">{name}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.warning(f"Автор `{uid_input}` не найден в базе.")

    # Segment descriptions
    st.markdown('<div class="section-header">Справка по сегментам</div>',
                unsafe_allow_html=True)
    descriptions = {
        0: "Регулярно пишут отзывы, высокий средний рейтинг (4.86), широкий охват брендов. Основная аудитория.",
        1: "Почти исключительно позитивные оценки (4.89), более узкий охват. Самые довольные авторы.",
        2: "Очень высокая активность (avg 20.7 отзывов), 17+ брендов. Ядро отзывного контента.",
        3: "Повышенная доля негативных оценок (15.4%), активны, rating 4.02. Критичные голоса.",
        4: "Умеренная активность, также критичны (rating 4.04, негатив 11.4%). Менее вовлечены.",
    }
    for sid, desc in descriptions.items():
        color = SEG_COLORS[sid]
        st.markdown(
            f'<div style="border-left:4px solid {color}; padding:0.6rem 1rem; '
            f'margin:0.4rem 0; background:white; border-radius:0 8px 8px 0">'
            f'<b>Сегмент {sid} — {SEG_NAMES[sid]}</b><br>'
            f'<span style="color:{GRAY_TEXT}; font-size:0.88rem">{desc}</span></div>',
            unsafe_allow_html=True,
        )
