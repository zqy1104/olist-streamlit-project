from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# =========================
# 基础配置
# =========================
st.set_page_config(
    page_title="Olist 电商履约与客户满意度分析平台",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
REPO_CLEAN_DIR = BASE_DIR / "clean"
DEFAULT_CLEAN_DIR = Path(r"G:\研究生\研一课程\信息分析与可视化\clean")
LOCAL_FALLBACK_CLEAN = Path("/mnt/data")


def resolve_data_dir() -> Path:
    candidates = [REPO_CLEAN_DIR, DEFAULT_CLEAN_DIR, LOCAL_FALLBACK_CLEAN]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return REPO_CLEAN_DIR


DATA_DIR = resolve_data_dir()

COLORS = {
    "bg": "#F4F6F8",
    "card": "#FFFFFF",
    "card_soft": "#FAFBFC",
    "ink": "#1F2937",
    "muted": "#6B7280",
    "line": "#D9E0E7",
    "primary": "#2F5D62",
    "secondary": "#6E8B8E",
    "accent": "#C2A46F",
    "accent_soft": "#D9C19A",
    "teal_soft": "#A9C1BE",
    "danger": "#9A4D4D",
    "danger_soft": "#D7B0B0",
    "ok": "#618B6F",
    "ok_soft": "#C4D8C8",
}

SEQUENCE = [
    COLORS["primary"], COLORS["secondary"], COLORS["accent"],
    COLORS["teal_soft"], COLORS["danger"], COLORS["accent_soft"], COLORS["ok"]
]
PLOT_TEMPLATE = "plotly_white"

PAYMENT_MAP = {
    "credit_card": "信用卡",
    "boleto": "Boleto",
    "voucher": "代金券",
    "debit_card": "借记卡",
    "not_defined": "未定义",
    "unknown": "未知",
}

FEATURE_MAP = {
    "is_delayed": "是否延期",
    "delay_days_positive_w": "延期天数",
    "purchase_to_approval_days_w": "下单到审核时长",
    "approval_to_carrier_days_w": "审核到物流交接时长",
    "transport_days_w": "物流运输时长",
    "total_fulfillment_days_w": "总履约时长",
    "freight_ratio_w": "运费占比",
    "item_count": "商品数",
    "seller_count": "卖家数",
    "seller_state_count": "卖家州数",
    "category_count": "品类数",
    "payment_installments_max": "分期数",
    "payment_value_total_w": "支付总金额",
    "weight_total_g_w": "总重量",
    "volume_total_cm3_w": "总体积",
    "order_complexity_index": "订单复杂度指数",
    "is_multi_seller": "是否多卖家",
}

COMPLEXITY_MAP = {
    "Q1 Low": "Q1 低复杂度",
    "Q2": "Q2",
    "Q3": "Q3",
    "Q4 High": "Q4 高复杂度",
}

QUADRANT_MAP = {
    "High delay / Low score": "高延期 / 低评分",
    "Low delay / High score": "低延期 / 高评分",
    "High delay / High score": "高延期 / 高评分",
    "Low delay / Low score": "低延期 / 低评分",
}

SELLER_MAP = {
    "Single seller": "单卖家订单",
    "Multi-seller": "多卖家订单",
}


# =========================
# 样式
# =========================
st.markdown(
    f"""
    <style>
    .stApp {{
        background: {COLORS['bg']};
        color: {COLORS['ink']};
    }}
    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1550px;
    }}
    h1, h2, h3 {{
        color: {COLORS['ink']};
        letter-spacing: .2px;
    }}
    .hero {{
        background: linear-gradient(135deg, #355E63 0%, #4D7478 55%, #7F9A9C 100%);
        border-radius: 22px;
        color: #ffffff;
        padding: 1.45rem 1.65rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 28px rgba(47,93,98,.16);
    }}
    .hero-title {{
        font-size: 1.95rem;
        font-weight: 700;
        margin-bottom: .35rem;
    }}
    .hero-subtitle {{
        font-size: .98rem;
        line-height: 1.72;
        opacity: .96;
    }}
    .card {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['line']};
        border-radius: 18px;
        padding: 1rem 1.08rem;
        box-shadow: 0 8px 24px rgba(31,41,55,.05);
        height: 100%;
    }}
    .module-card {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['line']};
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 24px rgba(31,41,55,.05);
        min-height: 170px;
    }}
    .module-card.active {{
        border: 1.2px solid {COLORS['primary']};
        box-shadow: 0 10px 28px rgba(47,93,98,.14);
    }}
    .module-label {{
        display: inline-block;
        font-size: .78rem;
        color: {COLORS['muted']};
        background: #EEF3F4;
        padding: .22rem .58rem;
        border-radius: 999px;
        margin-bottom: .68rem;
    }}
    .module-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {COLORS['ink']};
        margin-bottom: .45rem;
    }}
    .module-desc {{
        font-size: .92rem;
        color: {COLORS['muted']};
        line-height: 1.72;
    }}
    .current-pill {{
        display: inline-flex;
        align-items: center;
        gap: .42rem;
        background: #EEF3F4;
        color: {COLORS['primary']};
        border: 1px solid #D8E4E5;
        border-radius: 999px;
        padding: .34rem .82rem;
        font-size: .82rem;
        font-weight: 600;
        margin: .15rem 0 .85rem 0;
    }}
    .caption-note {{
        color: {COLORS['muted']};
        font-size: .88rem;
        line-height: 1.7;
        margin: .15rem 0 .85rem 0;
        padding: .55rem .8rem;
        background: #F7FAFB;
        border: 1px solid {COLORS['line']};
        border-radius: 12px;
    }}
    .kpi-card {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['line']};
        border-radius: 18px;
        padding: .95rem 1rem;
        box-shadow: 0 8px 24px rgba(31,41,55,.05);
        min-height: 108px;
    }}
    .kpi-title {{
        color: {COLORS['muted']};
        font-size: .82rem;
        margin-bottom: .35rem;
    }}
    .kpi-value {{
        font-size: 1.52rem;
        font-weight: 700;
        color: {COLORS['ink']};
    }}
    .kpi-note {{
        color: {COLORS['muted']};
        font-size: .78rem;
        margin-top: .2rem;
        line-height: 1.55;
    }}
    .section-title {{
        font-size: 1.12rem;
        font-weight: 700;
        color: {COLORS['ink']};
        margin: .1rem 0 .52rem 0;
    }}
    .section-note {{
        color: {COLORS['muted']};
        font-size: .93rem;
        line-height: 1.78;
    }}
    .insight-card {{
        background: linear-gradient(180deg, #FFFFFF 0%, #FAFBFC 100%);
        border: 1px solid {COLORS['line']};
        border-radius: 18px;
        padding: .9rem 1rem;
        min-height: 118px;
        box-shadow: 0 8px 24px rgba(31,41,55,.05);
    }}
    .insight-title {{
        font-size: .88rem;
        color: {COLORS['muted']};
        margin-bottom: .45rem;
    }}
    .insight-body {{
        font-size: .95rem;
        color: {COLORS['ink']};
        line-height: 1.72;
        font-weight: 600;
    }}
    [data-testid="stSidebar"] {{
        background: #F7F9FB;
        border-right: 1px solid {COLORS['line']};
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{
        background: #FFFFFF;
        border: 1px solid {COLORS['line']};
        border-radius: 12px;
        color: {COLORS['ink']};
        padding: 8px 14px;
    }}
    .stTabs [aria-selected="true"] {{
        background: #EEF3F4 !important;
        border-color: {COLORS['primary']} !important;
        color: {COLORS['primary']} !important;
        font-weight: 700;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# 数据读取
# =========================
@st.cache_data(show_spinner=False)
def load_data():
    files = {
        "orders": "olist_project_order_sample.csv",
        "state": "olist_state_summary.csv",
        "category": "olist_category_summary.csv",
        "feature": "olist_feature_importance.csv",
        "score_cost": "olist_score_cost_summary.csv",
        "complexity": "olist_complexity_summary.csv",
        "payment_type": "olist_payment_type_summary.csv",
        "seller_delay": "olist_seller_delay_summary.csv",
        "overview": "olist_project_overview.json",
        "model_metrics": "olist_model_metrics.json",
    }
    data = {}
    missing = []
    for key, filename in files.items():
        path = DATA_DIR / filename
        if not path.exists():
            missing.append(str(path))
            continue
        if filename.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data[key] = json.load(f)
        else:
            data[key] = pd.read_csv(path)
    return data, missing


def fmt_num(x, digits=2):
    if pd.isna(x):
        return "-"
    return f"{x:,.{digits}f}"


def fmt_pct(x, digits=1):
    if pd.isna(x):
        return "-"
    return f"{x:.{digits}f}%"


def kpi_card(title, value, note=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_card(title, body):
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">{title}</div>
            <div class="insight-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig, height=420):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        colorway=SEQUENCE,
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["ink"], size=13),
        margin=dict(l=28, r=22, t=58, b=30),
        legend_title_text="",
        hoverlabel=dict(bgcolor="white", font_size=12),
        height=height,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E9EEF2",
        zeroline=False,
        linecolor="#D7DEE5",
        tickfont=dict(color=COLORS["ink"]),
        title_font=dict(color=COLORS["ink"]),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E9EEF2",
        zeroline=False,
        linecolor="#D7DEE5",
        tickfont=dict(color=COLORS["ink"]),
        title_font=dict(color=COLORS["ink"]),
    )
    return fig


# =========================
# 派生分析函数
# =========================
@st.cache_data(show_spinner=False)
def prepare_orders(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    num_cols = [
        "review_score", "is_delayed", "delay_days_positive", "delay_days", "early_days", "is_low_score",
        "purchase_to_approval_days", "approval_to_carrier_days", "transport_days", "total_fulfillment_days",
        "freight_ratio", "item_count", "seller_count", "seller_state_count", "category_count",
        "payment_installments_max", "payment_value_total", "weight_total_g", "volume_total_cm3",
        "order_complexity_index", "is_multi_seller",
    ]
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["review_score"] = out["review_score"].round().astype("Int64")
    out["delay_bucket"] = pd.cut(
        out["delay_days_positive"].fillna(0),
        bins=[-0.1, 0, 1, 3, 7, 15, np.inf],
        labels=["准时", "延迟1天", "延迟2-3天", "延迟4-7天", "延迟8-15天", "延迟15天以上"]
    )
    out["delivery_status"] = np.where(out["is_delayed"] == 1, "延期送达", "准时送达")
    out["top_category"] = out.get("top_category", "unknown").astype(str).str.strip().replace({"": "unknown"})
    out["payment_type_cn"] = out.get("payment_type_primary", "unknown").map(PAYMENT_MAP).fillna("未知")
    if "complexity_level" in out.columns:
        out["complexity_level_cn"] = out["complexity_level"].map(COMPLEXITY_MAP).fillna(out["complexity_level"])
    else:
        out["complexity_level_cn"] = pd.qcut(
            out["order_complexity_index"].rank(method="first"),
            q=4,
            labels=["Q1 低复杂度", "Q2", "Q3", "Q4 高复杂度"],
        )
    out["seller_structure_cn"] = np.where(out["seller_count"].fillna(0) > 1, "多卖家订单", "单卖家订单")
    return out


@st.cache_data(show_spinner=False)
def get_filter_meta(orders: pd.DataFrame):
    states = sorted([x for x in orders["customer_state"].dropna().unique().tolist()])
    categories = sorted([str(x).strip() for x in orders["top_category"].dropna().astype(str).unique().tolist() if str(x).strip()])
    scores = sorted([int(x) for x in orders["review_score"].dropna().unique().tolist()])
    day_min = float(max(0, np.nanpercentile(orders["total_fulfillment_days"], 0)))
    day_max = float(np.nanpercentile(orders["total_fulfillment_days"], 99.5))
    return states, categories, scores, round(day_min, 1), round(max(day_max, day_min + 1), 1)


@st.cache_data(show_spinner=False)
def filter_orders(orders, states, categories, scores, delay_choice, day_range):
    out = orders.copy()
    if states:
        out = out[out["customer_state"].isin(states)]
    if categories:
        out["top_category"] = out["top_category"].astype(str).str.strip()
        sel_categories = [str(x).strip() for x in categories if str(x).strip()]
        out = out[out["top_category"].isin(sel_categories)]
    if scores:
        out = out[out["review_score"].isin(scores)]
    out = out[(out["total_fulfillment_days"] >= float(day_range[0])) & (out["total_fulfillment_days"] <= float(day_range[1]))]
    if delay_choice == "仅看准时":
        out = out[out["is_delayed"] == 0]
    elif delay_choice == "仅看延期":
        out = out[out["is_delayed"] == 1]
    return out


@st.cache_data(show_spinner=False)
def build_a_views(df):
    score_chain = (
        df.groupby("review_score")
        .agg(
            下单到审核=("purchase_to_approval_days", "mean"),
            审核到物流交接=("approval_to_carrier_days", "mean"),
            物流运输=("transport_days", "mean"),
            总履约时长=("total_fulfillment_days", "mean"),
            订单量=("order_id", "count"),
        )
        .reset_index()
    )

    chain_long = score_chain[["review_score", "下单到审核", "审核到物流交接", "物流运输"]].melt(
        id_vars="review_score", var_name="流程环节", value_name="平均时长"
    )

    score_total = score_chain[["review_score", "总履约时长", "订单量"]].copy()

    delay_curve = (
        df.groupby("delay_bucket", observed=False)
        .agg(低评分占比=("is_low_score", "mean"), 平均评分=("review_score", "mean"), 订单量=("order_id", "count"))
        .reset_index()
    )
    delay_curve["低评分占比"] = delay_curve["低评分占比"] * 100

    chain_compare = pd.DataFrame({
        "流程环节": ["下单到审核", "审核到物流交接", "物流运输"],
        "准时送达": [
            df.loc[df["is_delayed"] == 0, "purchase_to_approval_days"].mean(),
            df.loc[df["is_delayed"] == 0, "approval_to_carrier_days"].mean(),
            df.loc[df["is_delayed"] == 0, "transport_days"].mean(),
        ],
        "延期送达": [
            df.loc[df["is_delayed"] == 1, "purchase_to_approval_days"].mean(),
            df.loc[df["is_delayed"] == 1, "approval_to_carrier_days"].mean(),
            df.loc[df["is_delayed"] == 1, "transport_days"].mean(),
        ],
    })

    stage_rows = []
    stage_map = {
        "purchase_to_approval_days": "下单到审核",
        "approval_to_carrier_days": "审核到物流交接",
        "transport_days": "物流运输",
        "total_fulfillment_days": "总履约时长",
        "delay_days_positive": "延期天数",
    }
    for col, name in stage_map.items():
        temp = df[[col, "is_low_score", "review_score"]].dropna().copy()
        if len(temp) < 50 or temp[col].nunique() < 4:
            continue
        temp["quartile_bin"] = pd.qcut(
            temp[col].rank(method="first"),
            q=min(4, int(temp[col].nunique())),
            labels=False,
            duplicates="drop",
        )
        valid_bins = sorted(temp["quartile_bin"].dropna().unique().tolist())
        if len(valid_bins) < 2:
            continue
        q1 = temp[temp["quartile_bin"] == valid_bins[0]]
        q4 = temp[temp["quartile_bin"] == valid_bins[-1]]
        stage_rows.append({
            "环节": name,
            "低评分率提升(pp)": (q4["is_low_score"].mean() - q1["is_low_score"].mean()) * 100,
            "平均评分下降": q1["review_score"].mean() - q4["review_score"].mean(),
        })
    if stage_rows:
        stage_impact = pd.DataFrame(stage_rows).sort_values("低评分率提升(pp)", ascending=False)
    else:
        stage_impact = pd.DataFrame(columns=["环节", "低评分率提升(pp)", "平均评分下降"])

    delay_summary = (
        df.groupby("delivery_status")
        .agg(
            订单量=("order_id", "count"),
            平均评分=("review_score", "mean"),
            低评分占比=("is_low_score", "mean"),
            平均总履约时长=("total_fulfillment_days", "mean"),
            平均运输时长=("transport_days", "mean"),
        )
        .reset_index()
    )
    delay_summary["低评分占比"] = delay_summary["低评分占比"] * 100
    return chain_long, score_total, delay_curve, chain_compare, stage_impact, delay_summary


@st.cache_data(show_spinner=False)
def build_b_views(df):
    score_cost = (
        df.groupby("review_score")
        .agg(
            运费占比=("freight_ratio", "mean"),
            商品数=("item_count", "mean"),
            卖家数=("seller_count", "mean"),
            支付金额=("payment_value_total", "mean"),
            订单量=("order_id", "count"),
        )
        .reset_index()
    )
    score_cost["运费占比"] *= 100

    complexity = (
        df.groupby("complexity_level_cn")
        .agg(
            平均评分=("review_score", "mean"),
            低评分占比=("is_low_score", "mean"),
            延期率=("is_delayed", "mean"),
            平均总履约时长=("total_fulfillment_days", "mean"),
            平均运费占比=("freight_ratio", "mean"),
            订单量=("order_id", "count"),
        )
        .reset_index()
    )
    complexity["低评分占比"] *= 100
    complexity["延期率"] *= 100
    complexity["平均运费占比"] *= 100

    seller = (
        df.groupby("seller_structure_cn")
        .agg(
            平均评分=("review_score", "mean"),
            延期率=("is_delayed", "mean"),
            平均总履约时长=("total_fulfillment_days", "mean"),
            平均运输时长=("transport_days", "mean"),
            平均运费占比=("freight_ratio", "mean"),
            订单量=("order_id", "count"),
        )
        .reset_index()
    )
    seller["延期率"] *= 100
    seller["平均运费占比"] *= 100

    payment = (
        df.groupby("payment_type_cn")
        .agg(
            平均评分=("review_score", "mean"),
            低评分占比=("is_low_score", "mean"),
            平均支付金额=("payment_value_total", "mean"),
            平均分期数=("payment_installments_max", "mean"),
            订单量=("order_id", "count"),
        )
        .reset_index()
        .sort_values("订单量", ascending=False)
    )
    payment["低评分占比"] *= 100

    scatter_df = df[["weight_total_g", "volume_total_cm3", "total_fulfillment_days", "review_score", "seller_count", "item_count"]].dropna().copy()
    if len(scatter_df) > 2500:
        scatter_df = scatter_df.sample(2500, random_state=42)

    return score_cost, complexity, seller, payment, scatter_df


@st.cache_data(show_spinner=False)
def build_c_views(df):
    state_df = (
        df.groupby("customer_state")
        .agg(
            平均评分=("review_score", "mean"),
            低评分占比=("is_low_score", "mean"),
            延期率=("is_delayed", "mean"),
            平均总履约时长=("total_fulfillment_days", "mean"),
            平均运输时长=("transport_days", "mean"),
            平均运费占比=("freight_ratio", "mean"),
            订单量=("order_id", "count"),
        )
        .reset_index()
    )
    state_df["低评分占比"] *= 100
    state_df["延期率"] *= 100
    state_df["平均运费占比"] *= 100

    state_df["分层"] = np.select(
        [
            (state_df["延期率"] >= state_df["延期率"].mean()) & (state_df["平均评分"] < state_df["平均评分"].mean()),
            (state_df["延期率"] < state_df["延期率"].mean()) & (state_df["平均评分"] >= state_df["平均评分"].mean()),
            (state_df["延期率"] >= state_df["延期率"].mean()) & (state_df["平均评分"] >= state_df["平均评分"].mean()),
        ],
        ["高延期 / 低评分", "低延期 / 高评分", "高延期 / 高评分"],
        default="低延期 / 低评分",
    )

    cat_df = (
        df.groupby("top_category")
        .agg(
            平均评分=("review_score", "mean"),
            低评分占比=("is_low_score", "mean"),
            延期率=("is_delayed", "mean"),
            平均总履约时长=("total_fulfillment_days", "mean"),
            平均运费占比=("freight_ratio", "mean"),
            订单量=("order_id", "count"),
        )
        .reset_index()
        .sort_values("订单量", ascending=False)
    )
    cat_df["低评分占比"] *= 100
    cat_df["延期率"] *= 100
    cat_df["平均运费占比"] *= 100
    cat_top = cat_df.head(15).copy()
    return state_df, cat_df, cat_top


# =========================
# 读取数据
# =========================
data, missing = load_data()
if missing:
    st.error("以下结果文件未找到，请先运行预处理脚本：\n\n" + "\n".join(missing))
    st.stop()

orders = prepare_orders(data["orders"])
state_summary_raw = data["state"].copy()
category_summary_raw = data["category"].copy()
feature_importance = data["feature"].copy()
overview = data["overview"]
model_metrics = data["model_metrics"]

feature_importance["feature_cn"] = feature_importance["feature"].map(FEATURE_MAP).fillna(feature_importance["feature"])
state_summary_raw["quadrant_cn"] = state_summary_raw["quadrant"].map(QUADRANT_MAP).fillna(state_summary_raw["quadrant"])
category_summary_raw["avg_freight_ratio_proxy"] = pd.to_numeric(category_summary_raw["avg_freight_ratio_proxy"], errors="coerce") * 100

states, categories, score_options, min_days, max_days = get_filter_meta(orders)

if "flt_states" not in st.session_state:
    st.session_state.flt_states = []
if "flt_categories" not in st.session_state:
    st.session_state.flt_categories = []
if "flt_scores" not in st.session_state:
    st.session_state.flt_scores = score_options
if "flt_delay" not in st.session_state:
    st.session_state.flt_delay = "全部"
if "flt_days" not in st.session_state:
    st.session_state.flt_days = (min_days, max_days)

# =========================
# 顶部模块卡片
# =========================
MODULE_INFO = [
    {
        "key": "模块一：履约时效与流程体验",
        "label": "模块一",
        "title": "履约时效与流程体验",
        "desc": "以订单时间链为主线，识别下单、审核、物流交接、运输与延期对评分变化的影响，回答“满意度下降究竟发生在履约链条的哪一段”。",
    },
    {
        "key": "模块二：履约成本与订单复杂性",
        "label": "模块二",
        "title": "履约成本与订单复杂性",
        "desc": "从运费占比、商品数、卖家数、重量体积、分期数与支付金额等角度刻画订单复杂性，识别低满意订单的结构性来源。",
    },
    {
        "key": "模块三：地区与品类差异",
        "label": "模块三",
        "title": "地区与品类差异",
        "desc": "在州级空间差异与商品品类差异两个维度上，识别履约表现和满意度的异质性分布，并形成项目整体解释框架。",
    },
]

SECTION_DESC = {
    "项目总览": "展示订单规模、履约表现、成本结构与地区差异的整体轮廓，并作为三个分析模块的总入口。",
    "模块一：履约时效与流程体验": "沿订单履约链条展开分析，识别下单、审核、物流交接、运输与延期对客户满意度的差异化影响。",
    "模块二：履约成本与订单复杂性": "从订单成本、卖家结构、商品重量体积、支付方式与复杂度指数等角度识别低满意订单的结构性来源。",
    "模块三：地区与品类差异": "从州级空间差异与商品品类差异两个维度识别履约表现和满意度的异质性分布。",
    "综合结论": "整合三大模块的发现，概括履约体验、订单复杂性与地区品类差异对客户满意度的共同作用。",
}

def render_module_cards(active_section: str):
    cols = st.columns(3)
    active_key = active_section if active_section in {m["key"] for m in MODULE_INFO} else None
    for col, module in zip(cols, MODULE_INFO):
        active_cls = " active" if module["key"] == active_key else ""
        with col:
            st.markdown(
                f"""
                <div class="module-card{active_cls}">
                    <div class="module-label">{module['label']}</div>
                    <div class="module-title">{module['title']}</div>
                    <div class="module-desc">{module['desc']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

def render_current_section_badge(active_section: str):
    st.markdown(
        f"""
        <div class="current-pill">当前页面：{active_section}</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(SECTION_DESC.get(active_section, ""))

# =========================
# 页眉与模块结构
# =========================
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Olist 电商履约与客户满意度分析平台</div>
        <div class="hero-subtitle">
            本项目围绕“履约体验如何影响客户满意度”展开，按照订单履约链条、订单成本与结构复杂性、地区与品类差异三个层面建立统一分析框架。
            在方法上，将订单流程时长、成本结构、支付方式、卖家结构、地区与品类特征纳入同一订单级分析视角，既关注描述性差异，也强调影响路径与关键驱动因素识别。
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 页面导航")
    section = st.radio(
        "选择分析模块",
        ["项目总览", "模块一：履约时效与流程体验", "模块二：履约成本与订单复杂性", "模块三：地区与品类差异", "综合结论"],
    )

    st.markdown("### 分析范围")
    with st.form("filter_form"):
        selected_states = st.multiselect("客户州", options=states, default=st.session_state.flt_states)
        selected_categories = st.multiselect("主要品类", options=categories, default=st.session_state.flt_categories)
        selected_scores = st.multiselect("评分", options=score_options, default=st.session_state.flt_scores)
        selected_delay = st.radio("配送状态", ["全部", "仅看准时", "仅看延期"], index=["全部", "仅看准时", "仅看延期"].index(st.session_state.flt_delay))
        selected_days = st.slider("总履约时长（天）", min_value=float(min_days), max_value=float(max_days), value=(float(st.session_state.flt_days[0]), float(st.session_state.flt_days[1])), step=0.5)
        c_apply, c_reset = st.columns(2)
        with c_apply:
            submitted = st.form_submit_button("应用筛选", use_container_width=True)
        with c_reset:
            reset_clicked = st.form_submit_button("重置筛选", use_container_width=True)

if 'reset_clicked' not in locals():
    reset_clicked = False

if reset_clicked:
    st.session_state.flt_states = []
    st.session_state.flt_categories = []
    st.session_state.flt_scores = score_options
    st.session_state.flt_delay = "全部"
    st.session_state.flt_days = (min_days, max_days)
elif submitted:
    st.session_state.flt_states = selected_states
    st.session_state.flt_categories = selected_categories
    st.session_state.flt_scores = selected_scores
    st.session_state.flt_delay = selected_delay
    st.session_state.flt_days = selected_days

render_current_section_badge(section)
render_module_cards(section)

active_filters = []
if st.session_state.flt_states:
    active_filters.append(f"客户州：{len(st.session_state.flt_states)} 个")
if st.session_state.flt_categories:
    cat_preview = "，".join(st.session_state.flt_categories[:3])
    if len(st.session_state.flt_categories) > 3:
        cat_preview += f" 等 {len(st.session_state.flt_categories)} 个"
    active_filters.append(f"主要品类：{cat_preview}")
if st.session_state.flt_scores and len(st.session_state.flt_scores) < len(score_options):
    active_filters.append("评分：" + "、".join(map(str, st.session_state.flt_scores)))
if st.session_state.flt_delay != "全部":
    active_filters.append(f"配送状态：{st.session_state.flt_delay}")
if tuple(st.session_state.flt_days) != (min_days, max_days):
    active_filters.append(f"总履约时长：{st.session_state.flt_days[0]:.1f}–{st.session_state.flt_days[1]:.1f} 天")

if active_filters:
    st.markdown("<div class='caption-note'><b>当前筛选：</b>" + "；".join(active_filters) + "</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='caption-note'>当前筛选：全部样本</div>", unsafe_allow_html=True)

f_orders = filter_orders(
    orders,
    st.session_state.flt_states,
    st.session_state.flt_categories,
    st.session_state.flt_scores,
    st.session_state.flt_delay,
    st.session_state.flt_days,
)

if len(f_orders) == 0:
    st.warning("当前筛选条件下暂无数据。常见原因是：所选品类与履约时长区间、评分区间或州筛选叠加后样本为空。建议先将“总履约时长”恢复为全范围，再逐步缩小条件。")
    st.stop()

st.caption(f"当前样本量：{len(f_orders):,} 单")
chain_long, score_total, delay_curve, chain_compare, stage_impact, delay_summary = build_a_views(f_orders)
score_cost, complexity_view, seller_view, payment_view, scatter_view = build_b_views(f_orders)
state_view, category_view, category_top = build_c_views(f_orders)

# =========================
# 模块：项目总览
# =========================
if section == "项目总览":
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        kpi_card("订单样本规模", f"{overview['orders_delivered_with_review']:,}", "已完成配送且具有评分的订单")
    with k2:
        kpi_card("覆盖客户数", f"{overview['customers_covered']:,}", "进入项目分析框架的客户规模")
    with k3:
        kpi_card("平均评分", f"{overview['avg_review_score']:.2f}", "整体客户满意度基准")
    with k4:
        kpi_card("整体延期率", fmt_pct(overview['delayed_rate_pct']), "预计送达之后签收的订单占比")
    with k5:
        kpi_card("平均总履约时长", f"{overview['avg_total_fulfillment_days']:.2f} 天", "从下单到签收的平均耗时")
    with k6:
        kpi_card("平均运费占比", fmt_pct(overview['avg_freight_ratio_pct']), "运费占订单总成本的平均比重")

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("<div class='section-title'>整体研究框架</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="card">
                <div class="section-note">
                项目从订单级视角整合三条分析线索：<br>
                ① 履约时效：识别履约链条中最易触发评分下降的环节；<br>
                ② 成本与复杂性：解释订单本身的结构复杂程度和成本负担如何影响满意度；<br>
                ③ 地区与品类差异：说明履约体验与满意度在空间和业务结构上的异质性。<br><br>
                三个模块并非孤立展开，而是围绕同一问题递进：满意度下降究竟来自时效拖延、成本负担、复杂订单结构，还是地区与品类层面的系统性差异。
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("<div class='section-title'>评分分布概况</div>", unsafe_allow_html=True)
        score_dist = f_orders["review_score"].value_counts().sort_index().reset_index()
        score_dist.columns = ["评分", "订单量"]
        fig = px.bar(score_dist, x="评分", y="订单量", text_auto=True, color="评分", color_continuous_scale=[COLORS["teal_soft"], COLORS["primary"], COLORS["accent"]])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-title'>项目当前筛选范围下的时效链特征</div>", unsafe_allow_html=True)
        fig = px.bar(
            chain_long,
            x="review_score",
            y="平均时长",
            color="流程环节",
            barmode="stack",
            color_discrete_sequence=[COLORS["accent"], COLORS["teal_soft"], COLORS["primary"]],
        )
        fig.update_layout(xaxis_title="评分", yaxis_title="平均时长（天）")
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    with c2:
        st.markdown("<div class='section-title'>州级延期率与平均评分定位</div>", unsafe_allow_html=True)
        fig = px.scatter(
            state_view,
            x="延期率",
            y="平均评分",
            size="订单量",
            color="分层",
            hover_name="customer_state",
            color_discrete_map={
                "高延期 / 低评分": COLORS["danger"],
                "低延期 / 高评分": COLORS["ok"],
                "高延期 / 高评分": COLORS["accent"],
                "低延期 / 低评分": COLORS["secondary"],
            },
        )
        fig.add_vline(x=state_view["延期率"].mean(), line_dash="dot", line_color=COLORS["accent"])
        fig.add_hline(y=state_view["平均评分"].mean(), line_dash="dot", line_color=COLORS["accent"])
        fig.update_layout(xaxis_title="延期率（%）", yaxis_title="平均评分")
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    i1, i2, i3 = st.columns(3)
    transport_days = f_orders["transport_days"].mean()
    total_days = f_orders["total_fulfillment_days"].mean()
    share_transport = (transport_days / total_days * 100) if total_days and not pd.isna(total_days) else np.nan
    with i1:
        insight_card("主导耗时环节", f"当前样本中，物流运输平均耗时约为 {fmt_num(transport_days)} 天，占总履约时长约 {fmt_num(share_transport,1)}%，是履约链条中最主要的时间消耗段。")
    with i2:
        delayed_score = f_orders.loc[f_orders["is_delayed"] == 1, "review_score"].mean()
        ontime_score = f_orders.loc[f_orders["is_delayed"] == 0, "review_score"].mean()
        insight_card("延期的评分代价", f"延期订单平均评分为 {fmt_num(delayed_score)}，低于准时订单的 {fmt_num(ontime_score)}，说明满意度下降首先表现为准时性损失。")
    with i3:
        top_feature = feature_importance.sort_values("rf_importance", ascending=False).iloc[0]["feature_cn"] if not feature_importance.empty else "总履约时长"
        insight_card("复杂订单的关键驱动", f"在低满意订单识别模型中，{top_feature}位于重要性排序前列，说明成本和结构复杂性并非背景变量，而是满意度差异的重要来源。")

# =========================
# 模块一
# =========================
elif section == "模块一：履约时效与流程体验":
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        kpi_card("当前样本订单数", f"{len(f_orders):,}", "满足当前筛选条件的订单量")
    with k2:
        kpi_card("平均评分", f"{f_orders['review_score'].mean():.2f}", "当前样本的满意度水平")
    with k3:
        kpi_card("延期率", f"{f_orders['is_delayed'].mean():.1%}", "预计送达之后签收的订单占比")
    with k4:
        kpi_card("平均总履约时长", f"{f_orders['total_fulfillment_days'].mean():.2f} 天", "下单至签收的完整链路")
    with k5:
        kpi_card("平均运输时长", f"{f_orders['transport_days'].mean():.2f} 天", "物流运输段耗时")

    s1, s2, s3 = st.columns(3)
    with s1:
        insight_card("研究问题一", "订单是否准时送达，会不会直接改变客户评分分布？")
    with s2:
        insight_card("研究问题二", "履约链条中是审核慢、交接慢，还是运输慢，更容易拉低评分？")
    with s3:
        insight_card("研究问题三", "随着延期程度上升，低评分订单是否呈现持续抬升趋势？")

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("<div class='section-title'>订单流程时间链图</div>", unsafe_allow_html=True)
        fig = px.bar(
            chain_long,
            x="review_score",
            y="平均时长",
            color="流程环节",
            barmode="stack",
            color_discrete_sequence=[COLORS["accent"], COLORS["teal_soft"], COLORS["primary"]],
        )
        fig.update_layout(xaxis_title="评分", yaxis_title="平均时长（天）")
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    with a2:
        st.markdown("<div class='section-title'>不同评分组的平均总履约时长</div>", unsafe_allow_html=True)
        fig = px.bar(score_total, x="review_score", y="总履约时长", text_auto='.2f', color="总履约时长", color_continuous_scale=[COLORS["teal_soft"], COLORS["primary"], COLORS["accent"]])
        fig.update_layout(coloraxis_showscale=False, xaxis_title="评分", yaxis_title="平均总履约时长（天）")
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    a3, a4 = st.columns(2)
    with a3:
        st.markdown("<div class='section-title'>延期与未延期订单评分分布</div>", unsafe_allow_html=True)
        box_df = f_orders[["delivery_status", "review_score"]].dropna().copy()
        fig = px.box(
            box_df,
            x="delivery_status",
            y="review_score",
            color="delivery_status",
            points=False,
            color_discrete_map={"延期送达": COLORS["danger"], "准时送达": COLORS["primary"]},
        )
        fig.update_layout(xaxis_title="", yaxis_title="评分")
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    with a4:
        st.markdown("<div class='section-title'>延期天数与低评分占比变化</div>", unsafe_allow_html=True)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=delay_curve["delay_bucket"], y=delay_curve["低评分占比"], name="低评分占比", mode="lines+markers", line=dict(color=COLORS["danger"], width=3)), secondary_y=False)
        fig.add_trace(go.Bar(x=delay_curve["delay_bucket"], y=delay_curve["订单量"], name="订单量", marker_color=COLORS["teal_soft"], opacity=0.45), secondary_y=True)
        fig.update_xaxes(title_text="延期程度")
        fig.update_yaxes(title_text="低评分占比（%）", secondary_y=False)
        fig.update_yaxes(title_text="订单量", secondary_y=True)
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    a5, a6 = st.columns(2)
    with a5:
        st.markdown("<div class='section-title'>准时与延期订单的链路时长对比</div>", unsafe_allow_html=True)
        compare_long = chain_compare.melt(id_vars="流程环节", var_name="订单类型", value_name="平均时长")
        fig = px.bar(compare_long, x="流程环节", y="平均时长", color="订单类型", barmode="group", color_discrete_map={"准时送达": COLORS["primary"], "延期送达": COLORS["danger"]})
        fig.update_layout(xaxis_title="", yaxis_title="平均时长（天）")
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    with a6:
        st.markdown("<div class='section-title'>履约链条中哪一段更易拉低评分</div>", unsafe_allow_html=True)
        if stage_impact.empty:
            st.info("当前筛选样本较少，或各环节时长差异不足，暂无法稳定比较不同履约环节对低评分率的影响。建议放宽州、品类或评分筛选范围后再查看。")
        else:
            fig = px.bar(
                stage_impact,
                x="低评分率提升(pp)",
                y="环节",
                orientation="h",
                color="低评分率提升(pp)",
                color_continuous_scale=[COLORS["teal_soft"], COLORS["accent"], COLORS["danger"]],
                text_auto='.1f',
            )
            fig.update_layout(coloraxis_showscale=False, xaxis_title="从最快四分位到最慢四分位的低评分率提升（pp）", yaxis_title="")
            st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    st.markdown("<div class='section-title'>延期与准时订单对比摘要</div>", unsafe_allow_html=True)
    st.dataframe(
        delay_summary.style.format({"平均评分": "{:.2f}", "低评分占比": "{:.1f}%", "平均总履约时长": "{:.2f}", "平均运输时长": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

# =========================
# 模块二
# =========================
elif section == "模块二：履约成本与订单复杂性":
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        kpi_card("平均运费占比", fmt_pct(f_orders["freight_ratio"].mean() * 100), "运费在订单成本中的平均比重")
    with k2:
        kpi_card("平均商品数", f"{f_orders['item_count'].mean():.2f}", "每单包含的商品数量")
    with k3:
        kpi_card("平均卖家数", f"{f_orders['seller_count'].mean():.2f}", "每单涉及的卖家数量")
    with k4:
        kpi_card("平均支付金额", f"{f_orders['payment_value_total'].mean():.2f}", "订单实际支付金额")
    with k5:
        high_complex_rate = (f_orders["complexity_level_cn"] == "Q4 高复杂度").mean() * 100
        kpi_card("高复杂度订单占比", fmt_pct(high_complex_rate), "复杂度指数处于最高四分位")

    i1, i2, i3 = st.columns(3)
    with i1:
        insight_card("研究问题一", "运费占比越高，是否意味着客户评分更低？")
    with i2:
        insight_card("研究问题二", "多卖家、多商品、体积重量更大的订单，是否更容易拖慢履约体验？")
    with i3:
        insight_card("研究问题三", "低满意订单识别模型会把哪些变量排在最前面？")

    b1, b2 = st.columns(2)
    with b1:
        st.markdown("<div class='section-title'>评分组与运费占比关系</div>", unsafe_allow_html=True)
        fig = px.line(score_cost, x="review_score", y="运费占比", markers=True)
        fig.update_traces(line=dict(color=COLORS["accent"], width=3), marker=dict(size=9, color=COLORS["accent"]))
        fig.update_layout(xaxis_title="评分", yaxis_title="平均运费占比（%）")
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    with b2:
        st.markdown("<div class='section-title'>商品数与卖家数对评分的影响</div>", unsafe_allow_html=True)
        sample = f_orders[["review_score", "item_count", "seller_count"]].dropna().copy()
        if len(sample) > 4000:
            sample = sample.sample(4000, random_state=42)
        item_grp = sample.groupby("review_score").agg(商品数=("item_count", "mean"), 卖家数=("seller_count", "mean")).reset_index()
        item_long = item_grp.melt(id_vars="review_score", var_name="指标", value_name="平均值")
        fig = px.bar(item_long, x="review_score", y="平均值", color="指标", barmode="group", color_discrete_sequence=[COLORS["primary"], COLORS["accent"]])
        fig.update_layout(xaxis_title="评分", yaxis_title="平均值")
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    b3, b4 = st.columns(2)
    with b3:
        st.markdown("<div class='section-title'>重量与体积对履约时长的影响</div>", unsafe_allow_html=True)
        fig = px.scatter(
            scatter_view,
            x="weight_total_g",
            y="total_fulfillment_days",
            color="review_score",
            size="item_count",
            opacity=0.55,
            color_continuous_scale=[COLORS["teal_soft"], COLORS["primary"], COLORS["accent"]],
        )
        fig.update_layout(coloraxis_showscale=False, xaxis_title="总重量（g）", yaxis_title="总履约时长（天）")
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    with b4:
        st.markdown("<div class='section-title'>复杂度分组对比</div>", unsafe_allow_html=True)
        long_df = complexity_view.melt(id_vars="complexity_level_cn", value_vars=["平均评分", "低评分占比", "延期率"], var_name="指标", value_name="数值")
        fig = px.bar(long_df, x="complexity_level_cn", y="数值", color="指标", barmode="group", color_discrete_sequence=[COLORS["primary"], COLORS["danger"], COLORS["accent"]])
        fig.update_layout(xaxis_title="复杂度分组", yaxis_title="数值")
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    b5, b6 = st.columns(2)
    with b5:
        st.markdown("<div class='section-title'>多卖家订单与履约表现</div>", unsafe_allow_html=True)
        seller_long = seller_view.melt(id_vars="seller_structure_cn", value_vars=["平均评分", "延期率", "平均总履约时长"], var_name="指标", value_name="数值")
        fig = px.bar(seller_long, x="seller_structure_cn", y="数值", color="指标", barmode="group", color_discrete_sequence=[COLORS["primary"], COLORS["danger"], COLORS["accent"]])
        fig.update_layout(xaxis_title="订单结构", yaxis_title="数值")
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    with b6:
        st.markdown("<div class='section-title'>支付方式与满意度差异</div>", unsafe_allow_html=True)
        pay_show = payment_view.head(6).copy()
        fig = px.bar(pay_show, x="payment_type_cn", y="平均评分", color="平均评分", text_auto='.2f', color_continuous_scale=[COLORS["danger_soft"], COLORS["teal_soft"], COLORS["primary"]])
        fig.update_layout(coloraxis_showscale=False, xaxis_title="支付方式", yaxis_title="平均评分")
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    st.markdown("<div class='section-title'>低满意订单识别模型：特征重要性</div>", unsafe_allow_html=True)
    fi = feature_importance.sort_values("rf_importance", ascending=True).tail(12).copy()
    fig = px.bar(fi, x="rf_importance", y="feature_cn", orientation="h", color="rf_importance", color_continuous_scale=[COLORS["teal_soft"], COLORS["accent"], COLORS["danger"]], text_auto='.3f')
    fig.update_layout(coloraxis_showscale=False, xaxis_title="随机森林重要性", yaxis_title="")
    st.plotly_chart(style_fig(fig, 470), use_container_width=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        kpi_card("模型样本量", f"{model_metrics.get('sample_size', 0):,}", "进入低满意订单识别模型的订单量")
    with m2:
        kpi_card("随机森林 AUC", f"{model_metrics.get('rf_auc', 0):.3f}", "越接近 1 表示区分能力越强")
    with m3:
        kpi_card("Logistic AUC", f"{model_metrics.get('logit_auc', 0):.3f}", "用于验证结果稳定性")

# =========================
# 模块三
# =========================
elif section == "模块三：地区与品类差异":
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        kpi_card("覆盖州数", f"{state_view['customer_state'].nunique()}", "存在有效订单记录的州")
    with k2:
        kpi_card("覆盖品类数", f"{category_view['top_category'].nunique()}", "当前筛选范围下的主要品类数")
    with k3:
        kpi_card("州级平均延期率", fmt_pct(state_view['延期率'].mean()), "州层面的平均延期水平")
    with k4:
        kpi_card("州级平均评分", f"{state_view['平均评分'].mean():.2f}", "州层面的平均满意度")
    with k5:
        kpi_card("品类平均运费占比", fmt_pct(category_view['平均运费占比'].mean()), "品类层面的平均运费结构")

    i1, i2, i3 = st.columns(3)
    with i1:
        insight_card("研究问题一", "不同州的延期率和平均评分是否呈现明显分层？")
    with i2:
        insight_card("研究问题二", "哪些州处于高延期、低评分的重点关注区域？")
    with i3:
        insight_card("研究问题三", "哪些商品品类天然更容易出现高运费占比、长履约时长或低评分？")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-title'>州级延期率地理分布</div>", unsafe_allow_html=True)
        geo_src = data["state"].copy()
        geo_src = geo_src[geo_src["customer_state"].isin(state_view["customer_state"])].copy()
        fig = px.scatter_geo(
            geo_src,
            lat="lat",
            lon="lon",
            size="order_count",
            color="delayed_rate",
            hover_name="customer_state",
            projection="natural earth",
            color_continuous_scale=[COLORS["ok_soft"], COLORS["accent_soft"], COLORS["danger"]],
        )
        fig.update_geos(fitbounds="locations", visible=False, showcountries=True, countrycolor="#C9D2D8")
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), coloraxis_colorbar_title="延期率")
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    with c2:
        st.markdown("<div class='section-title'>州级平均评分地理分布</div>", unsafe_allow_html=True)
        geo_src2 = data["state"].copy()
        geo_src2 = geo_src2[geo_src2["customer_state"].isin(state_view["customer_state"])].copy()
        fig = px.scatter_geo(
            geo_src2,
            lat="lat",
            lon="lon",
            size="order_count",
            color="avg_review_score",
            hover_name="customer_state",
            projection="natural earth",
            color_continuous_scale=[COLORS["danger_soft"], COLORS["accent_soft"], COLORS["primary"]],
        )
        fig.update_geos(fitbounds="locations", visible=False, showcountries=True, countrycolor="#C9D2D8")
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), coloraxis_colorbar_title="平均评分")
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("<div class='section-title'>州级延期率与平均评分四象限</div>", unsafe_allow_html=True)
        fig = px.scatter(
            state_view,
            x="延期率",
            y="平均评分",
            size="订单量",
            color="分层",
            hover_name="customer_state",
            color_discrete_map={
                "高延期 / 低评分": COLORS["danger"],
                "低延期 / 高评分": COLORS["ok"],
                "高延期 / 高评分": COLORS["accent"],
                "低延期 / 低评分": COLORS["secondary"],
            },
        )
        fig.add_vline(x=state_view["延期率"].mean(), line_dash="dot", line_color=COLORS["accent"])
        fig.add_hline(y=state_view["平均评分"].mean(), line_dash="dot", line_color=COLORS["accent"])
        fig.update_layout(xaxis_title="延期率（%）", yaxis_title="平均评分")
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    with c4:
        st.markdown("<div class='section-title'>品类平均评分排名（Top 15）</div>", unsafe_allow_html=True)
        rank_df = category_top.sort_values("平均评分", ascending=True)
        fig = px.bar(rank_df, x="平均评分", y="top_category", orientation="h", color="平均评分", color_continuous_scale=[COLORS["danger_soft"], COLORS["accent_soft"], COLORS["primary"]])
        fig.update_layout(coloraxis_showscale=False, xaxis_title="平均评分", yaxis_title="")
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown("<div class='section-title'>品类延期率与运费占比对比（Top 12）</div>", unsafe_allow_html=True)
        show = category_top.head(12).copy()
        comp = show[["top_category", "延期率", "平均运费占比"]].melt(id_vars="top_category", var_name="指标", value_name="数值")
        fig = px.bar(comp, x="top_category", y="数值", color="指标", barmode="group", color_discrete_sequence=[COLORS["danger"], COLORS["accent"]])
        fig.update_layout(xaxis_title="品类", yaxis_title="数值", xaxis_tickangle=-35)
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    with c6:
        st.markdown("<div class='section-title'>品类运费占比—履约时长—低评分风险</div>", unsafe_allow_html=True)
        bubble = category_top.copy()
        fig = px.scatter(
            bubble,
            x="平均运费占比",
            y="平均总履约时长",
            size="订单量",
            color="低评分占比",
            hover_name="top_category",
            color_continuous_scale=[COLORS["teal_soft"], COLORS["accent"], COLORS["danger"]],
        )
        fig.update_layout(xaxis_title="平均运费占比（%）", yaxis_title="平均总履约时长（天）")
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

# =========================
# 综合结论
# =========================
else:
    st.markdown("<div class='section-title'>综合发现</div>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        transport_days = f_orders["transport_days"].mean()
        total_days = f_orders["total_fulfillment_days"].mean()
        share_transport = (transport_days / total_days * 100) if total_days and not pd.isna(total_days) else np.nan
        insight_card("结论一：履约链条中的关键瓶颈", f"满意度下降首先与时效损失相关，而在时效链条内部，物流运输段占总履约时长约 {fmt_num(share_transport,1)}%，是影响体验的核心环节。")
    with f2:
        multi = seller_view.set_index("seller_structure_cn") if not seller_view.empty else pd.DataFrame()
        if not multi.empty and {"多卖家订单", "单卖家订单"}.issubset(set(multi.index)):
            gap = multi.loc["多卖家订单", "延期率"] - multi.loc["单卖家订单", "延期率"]
            insight_card("结论二：复杂订单的结构性成本", f"多卖家订单的延期率相较单卖家订单高出约 {fmt_num(gap,1)} 个百分点，说明订单结构复杂性不仅增加协调成本，也会削弱满意度表现。")
        else:
            insight_card("结论二：复杂订单的结构性成本", "订单复杂性并非附属信息，而是满意度差异的重要来源，尤其体现在多卖家、高运费占比和高复杂度订单。")
    with f3:
        worst = state_view.sort_values(["延期率", "平均评分"], ascending=[False, True]).head(1)
        if len(worst) > 0:
            state_name = worst.iloc[0]["customer_state"]
            insight_card("结论三：空间与品类层面的异质性", f"地区与品类差异并非噪声。以 {state_name} 为代表的部分州同时表现出高延期和较低评分，说明履约问题具有明显空间分布特征。")
        else:
            insight_card("结论三：空间与品类层面的异质性", "地区与品类层面的履约表现和满意度并不一致，说明客户体验具有稳定的空间差异和业务结构差异。")

    st.markdown("<div class='section-title'>模块联动解读</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
            <div class="section-note">
            从整体上看，项目并不是分别讨论时效、成本和地区，而是形成了一条清晰的解释链：<br><br>
            1. 在订单流程层面，运输段拉长和延期发生后，客户满意度会明显下降；<br>
            2. 在订单结构层面，多卖家、高运费占比、高复杂度订单更容易累积履约摩擦，进而提高低评分风险；<br>
            3. 在地区和品类层面，这些履约摩擦并不是平均分布的，而会集中出现在部分州和部分品类中。<br><br>
            因此，客户满意度差异并不能仅用单一指标解释，而应理解为：履约时效、订单复杂性与地区品类差异共同作用下的结果。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("<div class='section-title'>重点关注州</div>", unsafe_allow_html=True)
        state_show = state_view.sort_values(["延期率", "平均评分"], ascending=[False, True]).head(10).copy()
        st.dataframe(state_show[["customer_state", "平均评分", "延期率", "平均总履约时长", "订单量", "分层"]].style.format({"平均评分": "{:.2f}", "延期率": "{:.1f}%", "平均总履约时长": "{:.2f}"}), use_container_width=True, hide_index=True)
    with t2:
        st.markdown("<div class='section-title'>重点关注品类</div>", unsafe_allow_html=True)
        cat_show = category_view.sort_values(["低评分占比", "平均总履约时长"], ascending=[False, False]).head(10).copy()
        st.dataframe(cat_show[["top_category", "平均评分", "低评分占比", "延期率", "平均总履约时长", "订单量"]].style.format({"平均评分": "{:.2f}", "低评分占比": "{:.1f}%", "延期率": "{:.1f}%", "平均总履约时长": "{:.2f}"}), use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>筛选后明细数据</div>", unsafe_allow_html=True)
    show_cols = [
        "order_id", "customer_state", "top_category", "review_score", "delivery_status", "delay_days_positive",
        "purchase_to_approval_days", "approval_to_carrier_days", "transport_days", "total_fulfillment_days",
        "freight_ratio", "item_count", "seller_count", "payment_value_total"
    ]
    display_df = f_orders[show_cols].copy().head(300)
    if "freight_ratio" in display_df.columns:
        display_df["freight_ratio"] = display_df["freight_ratio"] * 100
    st.dataframe(display_df.style.format({
        "purchase_to_approval_days": "{:.2f}", "approval_to_carrier_days": "{:.2f}", "transport_days": "{:.2f}",
        "total_fulfillment_days": "{:.2f}", "freight_ratio": "{:.1f}", "payment_value_total": "{:.2f}"
    }), use_container_width=True, hide_index=True)
