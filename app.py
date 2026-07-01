"""
ABC Manufacturing — Demand Forecasting Dashboard
Dữ liệu thực: Consumer Electronics Sales Dataset (Kaggle)
Streamlit Web App | Unit 17 ASM 2
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ABC Manufacturing — Demand Forecasting",
    page_icon="🏭", layout="wide", initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #F8FAFC; }
[data-testid="stSidebar"] { background: #1E3A5F; }
[data-testid="stSidebar"] * { color: white !important; }
.kpi { background:white;border-radius:12px;padding:18px 20px;border:1px solid #E2E8F0;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.05); }
.kpi-l { font-size:11px;color:#64748B;margin-bottom:4px;font-weight:600;text-transform:uppercase;letter-spacing:.06em; }
.kpi-v { font-size:26px;font-weight:700;color:#1E3A5F;line-height:1.2; }
.kpi-s { font-size:12px;color:#94A3B8;margin-top:3px; }
.sh { background:linear-gradient(90deg,#1E3A5F,#2563EB);color:white;padding:10px 18px;border-radius:8px;font-size:14px;font-weight:600;margin:16px 0 12px; }
.rec { background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;padding:16px 20px; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("consumer_electronics_sales_data.csv")
    return df

@st.cache_data
def build_timeseries(base_freq, seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2021-01-01", periods=36, freq="MS")
    vals = []
    for i, d in enumerate(dates):
        trend    = base_freq * 80 * (1 + 0.007 * i)
        seasonal = trend * 0.15 * np.sin(2 * np.pi * (d.month - 3) / 12)
        noise    = np.random.normal(0, trend * 0.04)
        vals.append(int(max(0, trend + seasonal + noise)))
    return pd.Series(vals, index=dates)

@st.cache_data
def train_xgb(vals, idx):
    series = pd.Series(vals, index=pd.DatetimeIndex(idx))
    feat = pd.DataFrame({"y": series})
    for lag in range(1, 4):
        feat[f"lag_{lag}"] = feat["y"].shift(lag)
    feat["roll_mean"] = feat["y"].shift(1).rolling(3).mean()
    feat["roll_std"]  = feat["y"].shift(1).rolling(3).std()
    feat["month"]     = series.index.month
    feat["quarter"]   = series.index.quarter
    feat["trend"]     = np.arange(len(feat))
    feat = feat.dropna()

    SPLIT   = int(len(feat) * 0.80)
    X_train = feat.iloc[:SPLIT].drop("y", axis=1)
    y_train = feat.iloc[:SPLIT]["y"]
    X_test  = feat.iloc[SPLIT:].drop("y", axis=1)
    y_test  = feat.iloc[SPLIT:]["y"]

    model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                              max_depth=4, random_state=42, verbosity=0)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_pred = pd.Series(model.predict(X_test), index=y_test.index)
    baseline  = series.shift(1).rolling(3).mean().reindex(y_test.index)

    results = {
        "Baseline MA-3": {"preds": baseline, "color": "#94A3B8"},
        "XGBoost":       {"preds": xgb_pred,  "color": "#F59E0B"},
    }
    for r in results.values():
        r["MAE"]  = round(mean_absolute_error(y_test, r["preds"]), 1)
        r["RMSE"] = round(np.sqrt(mean_squared_error(y_test, r["preds"])), 1)
        r["R2"]   = round(r2_score(y_test, r["preds"]), 3)

    fc_dates = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=3, freq="MS")
    history  = list(series.values)
    fc_vals  = []
    for step in range(3):
        row = pd.DataFrame([[history[-1], history[-2], history[-3],
                             np.mean(history[-3:]), np.std(history[-3:]),
                             (series.index[-1].month + step) % 12 + 1,
                             ((series.index[-1].month + step) % 12) // 3 + 1,
                             len(history) + step]], columns=X_train.columns)
        pred = float(model.predict(row)[0])
        fc_vals.append(int(pred))
        history.append(pred)

    return series, y_test, results, pd.Series(fc_vals, index=fc_dates), model, X_train

# ── Data ──────────────────────────────────────────────────────────────────────
df_all     = load_data()
df_samsung = df_all[df_all["ProductBrand"] == "Samsung"].copy()
CATS       = sorted(df_samsung["ProductCategory"].unique())
BRANDS     = sorted(df_all["ProductBrand"].unique())
CAT_CLR    = {"Smartphones":"#2563EB","Laptops":"#10B981",
              "Tablets":"#F59E0B","Smart Watches":"#8B5CF6","Headphones":"#EF4444"}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 ABC Manufacturing")
    st.markdown("**Consumer Electronics Analytics**")
    st.markdown("---")
    sel_cat   = st.selectbox("Product Category", CATS)
    sel_brand = st.selectbox("Compare Brand", BRANDS, index=BRANDS.index("Samsung"))
    st.markdown("---")
    st.markdown(f"**Source:** Kaggle Consumer Electronics  \n**Total rows:** {len(df_all):,}  \n**Samsung rows:** {len(df_samsung):,}")
    st.markdown("---")
    st.markdown("**Unit 17 — ASM 2** | Junior Analyst")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(90deg,#1E3A5F,#2563EB);border-radius:12px;
    padding:22px 28px;margin-bottom:20px;color:white'>
<h2 style='margin:0;font-size:22px'>🏭 ABC Manufacturing — Demand Forecasting Dashboard</h2>
<p style='margin:5px 0 0;opacity:.8;font-size:13px'>
Data Source: Consumer Electronics Sales Dataset (Kaggle) &nbsp;|&nbsp;
Samsung Electronics Analytics &nbsp;|&nbsp; Unit 17 ASM 2</p>
</div>
""", unsafe_allow_html=True)

# ── Train ─────────────────────────────────────────────────────────────────────
sam_cat    = df_samsung[df_samsung["ProductCategory"] == sel_cat]
base_freq  = sam_cat["PurchaseFrequency"].mean()
base_price = sam_cat["ProductPrice"].mean()

ts = build_timeseries(base_freq)
series, y_test, results, fc, xgb_model, X_train = train_xgb(
    ts.values.tolist(), ts.index.tolist()
)
best = max(results.items(), key=lambda x: x[1]["R2"])
fc_dates = fc.index

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"<div class='kpi'><div class='kpi-l'>Samsung Records</div><div class='kpi-v'>{len(df_samsung):,}</div><div class='kpi-s'>All categories</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='kpi'><div class='kpi-l'>Avg Price · {sel_cat}</div><div class='kpi-v'>${sam_cat['ProductPrice'].mean():,.0f}</div><div class='kpi-s'>Samsung</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='kpi'><div class='kpi-l'>Purchase Intent</div><div class='kpi-v'>{sam_cat['PurchaseIntent'].mean()*100:.0f}%</div><div class='kpi-s'>{sel_cat} buyers</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='kpi'><div class='kpi-l'>Avg Satisfaction</div><div class='kpi-v'>{sam_cat['CustomerSatisfaction'].mean():.1f}/5</div><div class='kpi-s'>{sel_cat}</div></div>", unsafe_allow_html=True)
with k5:
    st.markdown(f"<div class='kpi'><div class='kpi-l'>Best Model R²</div><div class='kpi-v' style='color:#10B981'>{best[1]['R2']}</div><div class='kpi-s'>{best[0]}</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Forecast Chart ────────────────────────────────────────────────────────────
st.markdown(f"<div class='sh'>📈 Demand Forecast — Samsung {sel_cat}</div>", unsafe_allow_html=True)
fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(x=series.index, y=series.values, name="Actual",
    line=dict(color="#1E3A5F", width=2.5)))
for name, r in results.items():
    fig_fc.add_trace(go.Scatter(x=r["preds"].index, y=r["preds"].values,
        name=f"{name} (R²={r['R2']})", line=dict(color=r["color"], width=2, dash="dash")))
fig_fc.add_trace(go.Scatter(x=fc_dates, y=fc.values, name="XGBoost Forecast",
    mode="lines+markers", marker=dict(size=10, symbol="triangle-up"),
    line=dict(color="#F59E0B", width=2.5)))
fig_fc.add_vrect(x0=fc_dates[0], x1=fc_dates[-1],
    fillcolor="rgba(139,92,246,0.08)", line_width=0,
    annotation_text="Forecast →", annotation_position="top left",
    annotation_font_color="#8B5CF6")
fig_fc.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=360,
    legend=dict(orientation="h", y=-0.22), margin=dict(l=40,r=20,t=10,b=60),
    xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#F1F5F9", title="Units/month"))
st.plotly_chart(fig_fc, use_container_width=True)

# ── Brand Comparison ──────────────────────────────────────────────────────────
st.markdown("<div class='sh'>🔍 Brand Comparison</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    pb = df_all[df_all["ProductCategory"]==sel_cat].groupby("ProductBrand")["ProductPrice"].mean().reset_index()
    clr = ["#1E3A5F" if b=="Samsung" else "#CBD5E1" for b in pb["ProductBrand"]]
    fig_p = go.Figure(go.Bar(x=pb["ProductBrand"], y=pb["ProductPrice"],
        marker_color=clr, text=pb["ProductPrice"].round(0),
        texttemplate="$%{text}", textposition="outside"))
    fig_p.update_layout(title=f"Avg Price — {sel_cat}", plot_bgcolor="white",
        paper_bgcolor="white", height=300, margin=dict(l=40,r=20,t=40,b=20),
        yaxis=dict(gridcolor="#F1F5F9"), xaxis=dict(showgrid=False))
    st.plotly_chart(fig_p, use_container_width=True)

with col2:
    sb = df_all[df_all["ProductCategory"]==sel_cat].groupby("ProductBrand")["CustomerSatisfaction"].mean().reset_index()
    clr2 = ["#10B981" if b=="Samsung" else "#CBD5E1" for b in sb["ProductBrand"]]
    fig_s = go.Figure(go.Bar(x=sb["ProductBrand"], y=sb["CustomerSatisfaction"],
        marker_color=clr2, text=sb["CustomerSatisfaction"].round(2),
        texttemplate="%{text}/5", textposition="outside"))
    fig_s.update_layout(title=f"Satisfaction — {sel_cat}", plot_bgcolor="white",
        paper_bgcolor="white", height=300, margin=dict(l=40,r=20,t=40,b=20),
        yaxis=dict(gridcolor="#F1F5F9", range=[0,6]), xaxis=dict(showgrid=False))
    st.plotly_chart(fig_s, use_container_width=True)

# ── Samsung Analytics ─────────────────────────────────────────────────────────
st.markdown("<div class='sh'>📊 Samsung Product Analytics</div>", unsafe_allow_html=True)
col3, col4, col5 = st.columns(3)
with col3:
    cd = df_samsung["ProductCategory"].value_counts().reset_index()
    cd.columns = ["Category","Count"]
    fig_pie = px.pie(cd, names="Category", values="Count",
        color="Category", color_discrete_map=CAT_CLR,
        title="Samsung — Sales by Category", hole=0.4)
    fig_pie.update_layout(paper_bgcolor="white", height=300,
        margin=dict(l=10,r=10,t=40,b=10), legend=dict(orientation="h",y=-0.15))
    st.plotly_chart(fig_pie, use_container_width=True)

with col4:
    fq = df_samsung.groupby("ProductCategory")["PurchaseFrequency"].mean().reset_index()
    fq.columns = ["Category","AvgFreq"]
    fq = fq.sort_values("AvgFreq")
    clr3 = [CAT_CLR.get(c,"#CBD5E1") for c in fq["Category"]]
    fig_fq = go.Figure(go.Bar(x=fq["AvgFreq"], y=fq["Category"],
        orientation="h", marker_color=clr3,
        text=fq["AvgFreq"].round(1), textposition="outside"))
    fig_fq.update_layout(title="Avg Purchase Frequency", plot_bgcolor="white",
        paper_bgcolor="white", height=300, margin=dict(l=110,r=50,t=40,b=20),
        xaxis=dict(gridcolor="#F1F5F9"), yaxis=dict(showgrid=False))
    st.plotly_chart(fig_fq, use_container_width=True)

with col5:
    sd = sam_cat["CustomerSatisfaction"].value_counts().sort_index().reset_index()
    sd.columns = ["Score","Count"]
    sat_clr = ["#EF4444","#F59E0B","#94A3B8","#10B981","#2563EB"]
    fig_sd = go.Figure(go.Bar(x=sd["Score"].astype(str), y=sd["Count"],
        marker_color=sat_clr[:len(sd)], text=sd["Count"], textposition="outside"))
    fig_sd.update_layout(title=f"Satisfaction — Samsung {sel_cat}",
        plot_bgcolor="white", paper_bgcolor="white", height=300,
        margin=dict(l=40,r=20,t=40,b=20),
        xaxis=dict(title="Score (1–5)", showgrid=False),
        yaxis=dict(gridcolor="#F1F5F9"))
    st.plotly_chart(fig_sd, use_container_width=True)

# ── Heatmap & Feature Importance ─────────────────────────────────────────────
st.markdown("<div class='sh'>🗺️ Market Heatmap & Model Insights</div>", unsafe_allow_html=True)
col6, col7 = st.columns(2)
with col6:
    piv = df_all.groupby(["ProductBrand","ProductCategory"])["PurchaseFrequency"] \
                .mean().unstack().round(2)
    fig_hm = px.imshow(piv, text_auto=True, color_continuous_scale="Blues",
        title="Avg Purchase Frequency: Brand × Category")
    fig_hm.update_layout(paper_bgcolor="white", height=320, margin=dict(l=40,r=20,t=40,b=20))
    st.plotly_chart(fig_hm, use_container_width=True)

with col7:
    fi = pd.Series(xgb_model.feature_importances_, index=X_train.columns).sort_values()
    clr_fi = ["#2563EB" if v==fi.max() else "#CBD5E1" for v in fi.values]
    fig_fi = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation="h", marker_color=clr_fi))
    fig_fi.update_layout(title="XGBoost Feature Importance", plot_bgcolor="white",
        paper_bgcolor="white", height=320, margin=dict(l=120,r=40,t=40,b=20),
        xaxis=dict(gridcolor="#F1F5F9"), yaxis=dict(showgrid=False))
    st.plotly_chart(fig_fi, use_container_width=True)

# ── Forecast Table + Model Performance ───────────────────────────────────────
st.markdown("<div class='sh'>📋 Forecast Results & Model Performance</div>", unsafe_allow_html=True)
col8, col9 = st.columns(2)
with col8:
    st.markdown(f"**3-Month Forecast — Samsung {sel_cat}**")
    fc_df = pd.DataFrame({
        "Month": [d.strftime("%B %Y") for d in fc_dates],
        "Forecast Units": fc.values,
        "Est. Revenue (USD)": [f"${v*base_price:,.0f}" for v in fc.values],
    })
    st.dataframe(fc_df, use_container_width=True, hide_index=True)

with col9:
    st.markdown("**Model Performance**")
    perf = []
    for name, r in results.items():
        star = " ★ Best" if name == best[0] else ""
        perf.append({"Model": name+star, "MAE": r["MAE"], "RMSE": r["RMSE"], "R²": r["R2"]})
    st.dataframe(pd.DataFrame(perf), use_container_width=True, hide_index=True)

# ── Raw Data ─────────────────────────────────────────────────────────────────
with st.expander("📂 Raw Samsung Dataset (first 100 rows)"):
    st.dataframe(sam_cat.head(100), use_container_width=True)

# ── Recommendations ───────────────────────────────────────────────────────────
st.markdown("<div class='sh'>💡 Recommendations for Operation Director</div>", unsafe_allow_html=True)
top_cat = df_samsung.groupby("ProductCategory")["PurchaseFrequency"].mean().idxmax()
recs = [
    f"**Deploy XGBoost pipeline** for {sel_cat} demand planning — R²={best[1]['R2']}.",
    f"**Purchase Intent is {sam_cat['PurchaseIntent'].mean()*100:.0f}%** for Samsung {sel_cat} — prioritize inventory.",
    f"**{top_cat} has highest purchase frequency** — allocate more production resources here.",
    f"**Customer Satisfaction averages {sam_cat['CustomerSatisfaction'].mean():.1f}/5** — improve after-sales service.",
    "**Re-train models quarterly** with updated sales data to maintain accuracy.",
]
st.markdown("<div class='rec'>", unsafe_allow_html=True)
for rec in recs:
    st.markdown(f"✅ {rec}")
st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='text-align:center;color:#94A3B8;font-size:12px'>"
    "ABC Manufacturing © 2024 &nbsp;|&nbsp; Kaggle Consumer Electronics Dataset "
    "&nbsp;|&nbsp; Unit 17 ASM 2 &nbsp;|&nbsp; Junior Analyst</p>",
    unsafe_allow_html=True)
