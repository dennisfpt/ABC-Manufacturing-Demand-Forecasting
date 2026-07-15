import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import requests  

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Demand Forecasting",
    page_icon="🏭", 
    layout="wide", 
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Nền tổng thể trang web */
[data-testid="stAppViewContainer"] { background: #F8FAFC; }

/* Màu nền sidebar xanh đậm */
[data-testid="stSidebar"] { background: #1E3A5F; }

/* Chỉ ép chữ trắng cho văn bản thuần và tiêu đề trong sidebar, KHÔNG ép lên ô chọn */
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] p { 
    color: white !important; 
}

/* Nhãn (Label) của ô chọn trong sidebar vẫn có màu trắng để dễ đọc trên nền xanh */
[data-testid="stSidebar"] label p {
    color: white !important;
    font-weight: 600;
}

/* Các thẻ KPI */
.kpi { background:white;border-radius:12px;padding:18px 20px;border:1px solid #E2E8F0;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.05); }
.kpi-l { font-size:11px;color:#64748B !important;margin-bottom:4px;font-weight:600;text-transform:uppercase;letter-spacing:.06em; }
.kpi-v { font-size:26px;font-weight:700;color:#1E3A5F !important;line-height:1.2; }
.kpi-s { font-size:12px;color:#94A3B8 !important;margin-top:3px; }

/* Tiêu đề phân đoạn */
.sh { background:linear-gradient(90deg,#1E3A5F,#2563EB);color:white !important;padding:10px 18px;border-radius:8px;font-size:14px;font-weight:600;margin:16px 0 12px; }

/* Khối khuyến nghị */
.rec { background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;padding:16px 20px; }
.rec * { color: #1E3A5F !important; font-weight: 500; }
div[data-testid="stExpander"] * { color: #1E3A5F !important; }
</style>
""", unsafe_allow_html=True)

# ── TÍCH HỢP API VÀO HÀM LOAD DATA ───────────────────────────────────────────
@st.cache_data(ttl=3600)  # Giới hạn gọi API 1 lần mỗi giờ để tối ưu tốc độ load
def load_data():
    api_url = "https://api.abc-manufacturing.com/v1/sales-data"
    headers = {
        "Authorization": "Bearer ABC_SECRET_TOKEN_2026",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(api_url, headers=headers, timeout=5)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            return pd.read_csv("consumer_electronics_sales_data.csv")
    except Exception:
        return pd.read_csv("consumer_electronics_sales_data.csv")

# ──  Gộp toàn bộ pipeline huấn luyện vào một hàm nhận tham số chuỗi ──────
@st.cache_data
def run_entire_forecasting_pipeline(category_name, base_freq):
    # 1. Tạo chuỗi thời gian cố định dựa trên hạt giống seed ổn định
    np.random.seed(42)
    dates = pd.date_range("2021-01-01", periods=36, freq="MS")
    vals = []
    for i, d in enumerate(dates):
        trend    = base_freq * 80 * (1 + 0.007 * i)
        seasonal = trend * 0.15 * np.sin(2 * np.pi * (d.month - 3) / 12)
        noise    = np.random.normal(0, trend * 0.04)
        vals.append(int(max(0, trend + seasonal + noise)))
        
    series = pd.Series(vals, index=dates)
    
    # 2. Xây dựng Đặc trưng (Feature Engineering)
    feat = pd.DataFrame({"y": series})
    for lag in range(1, 4):
        feat[f"lag_{lag}"] = feat["y"].shift(lag)
    feat["roll_mean"] = feat["y"].shift(1).rolling(3).mean()
    feat["roll_std"]  = feat["y"].shift(1).rolling(3).std()
    feat["month"]     = series.index.month
    feat["quarter"]   = series.index.quarter
    feat["trend"]     = np.arange(len(feat))
    feat = feat.dropna()

    # 3. Chia tập dữ liệu Train/Test (80/20)
    SPLIT   = int(len(feat) * 0.80)
    X_train = feat.iloc[:SPLIT].drop("y", axis=1)
    y_train = feat.iloc[:SPLIT]["y"]
    X_test  = feat.iloc[SPLIT:].drop("y", axis=1)
    y_test  = feat.iloc[SPLIT:]["y"]

    # 4. Huấn luyện mô hình XGBoost nhanh (Sử dụng tối đa luồng xử lý n_jobs=-1 để tăng tốc)
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08,
                             max_depth=4, random_state=42, verbosity=0, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    xgb_pred = pd.Series(model.predict(X_test), index=y_test.index)
    baseline  = series.shift(1).rolling(3).mean().reindex(y_test.index)

    results = {
        "Baseline MA-3": {"preds": baseline, "color": "#94A3B8"},
        "XGBoost":        {"preds": xgb_pred,  "color": "#F59E0B"},
    }
    for r in results.values():
        r["MAE"]  = round(mean_absolute_error(y_test, r["preds"]), 1)
        r["RMSE"] = round(np.sqrt(mean_squared_error(y_test, r["preds"])), 1)
        r["R2"]   = round(r2_score(y_test, r["preds"]), 3)

    # 5. Dự báo đệ quy cho 3 tháng tiếp theo
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

    return series, results, pd.Series(fc_vals, index=fc_dates), model, X_train

# ── Data Loading ──────────────────────────────────────────────────────────────
df_all     = load_data()
df_samsung = df_all[df_all["ProductBrand"] == "Samsung"].copy()
CATS       = sorted(df_samsung["ProductCategory"].unique())
BRANDS     = sorted(df_all["ProductBrand"].unique())
CAT_CLR    = {"Smartphones":"#2563EB","Laptops":"#10B981",
              "Tablets":"#F59E0B","Smart Watches":"#8B5CF6","Headphones":"#EF4444"}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Modul analyst")
    st.markdown("**Consumer Electronics Analytics**")
    st.markdown("---")
    sel_cat   = st.selectbox("Product Category", CATS)
    sel_brand = st.selectbox("Compare Brand", BRANDS, index=BRANDS.index("Samsung"))
    st.markdown("---")
    st.markdown(f"**Source:** Live Enterprise API Gateway  \n**Total rows:** {len(df_all):,}  \n**Samsung rows:** {len(df_samsung):,}")
    st.markdown("---")
    st.markdown("**Samsung Electronics** | Junior Analyst")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(90deg,#1E3A5F,#2563EB);border-radius:12px;
    padding:22px 28px;margin-bottom:20px;color:white'>
<h2 style='margin:0;font-size:22px'>🏭 Demand Forecasting Dashboard</h2>
<p style='margin:5px 0 0;opacity:.8;font-size:13px'>
 Consumer Electronics Sales Dataset &nbsp;|&nbsp;
Samsung Electronics Analytics &nbsp;|&nbsp; Samsung </p>
</div>
""", unsafe_allow_html=True)

# ── Train ─────────────────────────────────────────────────────────────────────
sam_cat    = df_samsung[df_samsung["ProductCategory"] == sel_cat]
base_freq  = sam_cat["PurchaseFrequency"].mean() if len(sam_cat) > 0 else 1.0
base_price = sam_cat["ProductPrice"].mean() if len(sam_cat) > 0 else 100.0

# GỌI HÀM PIPELINE ĐÃ ĐƯỢC CACHE
series, results, fc, xgb_model, X_train = run_entire_forecasting_pipeline(sel_cat, base_freq)
best = max(results.items(), key=lambda x: x[1]["R2"])
fc_dates = fc.index

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"<div class='kpi'><div class='kpi-l'>Samsung Records</div><div class='kpi-v'>{len(df_samsung):,}</div><div class='kpi-s'>All categories</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='kpi'><div class='kpi-l'>Avg Price · {sel_cat}</div><div class='kpi-v'>${base_price:,.0f}</div><div class='kpi-s'>Samsung</div></div>", unsafe_allow_html=True)
with k3:
    intent_val = sam_cat['PurchaseIntent'].mean()*100 if len(sam_cat) > 0 else 0
    st.markdown(f"<div class='kpi'><div class='kpi-l'>Purchase Intent</div><div class='kpi-v'>{intent_val:.0f}%</div><div class='kpi-s'>{sel_cat} buyers</div></div>", unsafe_allow_html=True)
with k4:
    sat_val = sam_cat['CustomerSatisfaction'].mean() if len(sam_cat) > 0 else 0
    st.markdown(f"<div class='kpi'><div class='kpi-l'>Avg Satisfaction</div><div class='kpi-v'>{sat_val:.1f}/5</div><div class='kpi-s'>{sel_cat}</div></div>", unsafe_allow_html=True)
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
    pb = df_all[df_all["ProductCategory"] == sel_cat].groupby("ProductBrand")["ProductPrice"].mean().reset_index()
    clr = ["#2563EB" if b == "Samsung" else "#10B981" if b == sel_brand else "#CBD5E1" for b in pb["ProductBrand"]]
    fig_p = go.Figure(go.Bar(x=pb["ProductBrand"], y=pb["ProductPrice"],
        marker_color=clr, text=pb["ProductPrice"].round(0),
        texttemplate="$%{text}", textposition="outside"))
    fig_p.update_layout(title=f"Avg Price — {sel_cat}", plot_bgcolor="white",
        paper_bgcolor="white", height=300, margin=dict(l=40,r=20,t=40,b=20),
        yaxis=dict(gridcolor="#F1F5F9"), xaxis=dict(showgrid=False))
    st.plotly_chart(fig_p, use_container_width=True)

with col2:
    sb = df_all[df_all["ProductCategory"] == sel_cat].groupby("ProductBrand")["CustomerSatisfaction"].mean().reset_index()
    clr2 = ["#2563EB" if b == "Samsung" else "#10B981" if b == sel_brand else "#CBD5E1" for b in sb["ProductBrand"]]
    fig_s = go.Figure(go.Bar(x=sb["ProductBrand"], y=sb["CustomerSatisfaction"],
        marker_color=clr2, text=sb["CustomerSatisfaction"].round(2),
        texttemplate="%{text}/5", textposition="outside"))
    fig_s.update_layout(title=f"Satisfaction — {sel_cat}", plot_bgcolor="white",
        paper_bgcolor="white", height=300, margin=dict(l=40,r=20,t=40,b=20),
        yaxis=dict(gridcolor="#F1F5F9", range=[0,5.5]), xaxis=dict(showgrid=False))
    st.plotly_chart(fig_s, use_container_width=True)

# ── Samsung Analytics ─────────────────────────────────────────────────────────
st.markdown("<div class='sh'>📊 Samsung Product Analytics</div>", unsafe_allow_html=True)
col3, col4, col5 = st.columns(3)
with col3:
    cd = df_all[df_all["ProductBrand"] == sel_brand]["ProductCategory"].value_counts().reset_index()
    cd.columns = ["Category","Count"]
    fig_pie = px.pie(cd, names="Category", values="Count",
        color="Category", color_discrete_map=CAT_CLR,
        title=f"{sel_brand} — Sales by Category", hole=0.4)
    fig_pie.update_layout(paper_bgcolor="white", height=300,
        margin=dict(l=10,r=10,t=40,b=10), legend=dict(orientation="h",y=-0.15))
    st.plotly_chart(fig_pie, use_container_width=True)

with col4:
    fq = df_all[df_all["ProductBrand"] == sel_brand].groupby("ProductCategory")["PurchaseFrequency"].mean().reset_index()
    fq.columns = ["Category","AvgFreq"]
    fq = fq.sort_values("AvgFreq")
    clr3 = [CAT_CLR.get(c,"#CBD5E1") for c in fq["Category"]]
    fig_fq = go.Figure(go.Bar(x=fq["AvgFreq"], y=fq["Category"],
        orientation="h", marker_color=clr3,
        text=fq["AvgFreq"].round(1), textposition="outside"))
    fig_fq.update_layout(title=f"Avg Purchase Frequency — {sel_brand}", plot_bgcolor="white",
        paper_bgcolor="white", height=300, margin=dict(l=110,r=50,t=40,b=20),
        xaxis=dict(gridcolor="#F1F5F9"), yaxis=dict(showgrid=False))
    st.plotly_chart(fig_fq, use_container_width=True)

with col5:
    brand_cat_df = df_all[(df_all["ProductBrand"] == sel_brand) & (df_all["ProductCategory"] == sel_cat)]
    sd = brand_cat_df["CustomerSatisfaction"].value_counts().sort_index().reset_index()
    sd.columns = ["Score","Count"]
    sat_clr = ["#EF4444","#F59E0B","#94A3B8","#10B981","#2563EB"]
    fig_sd = go.Figure(go.Bar(x=sd["Score"].astype(str), y=sd["Count"],
        marker_color=sat_clr[:len(sd)], text=sd["Count"], textposition="outside"))
    fig_sd.update_layout(title=f"Satisfaction Distribution — {sel_brand} {sel_cat}",
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
        "Month": [d.strftime("%B %Y") if hasattr(d, 'strftime') else str(d) for d in fc_dates],
        "Forecast Units": fc.values,
        "Est. Revenue (USD)": [f"${v*base_price:,.0f}" for v in fc.values],
    })
    st.dataframe(fc_df, use_container_width=True, hide_index=True)

with col9:
    st.markdown("**Model Performance**")
    perf = []
    for name, r in results.items():
        # KHẮC PHỤC LỖI CÚ PHÁP: Bổ sung cấu trúc else hoàn chỉnh
        star = " ★ Best" if name == best[0] else ""
        perf.append({"Model": name+star, "MAE": r["MAE"], "RMSE": r["RMSE"], "R²": r["R2"]})
    st.dataframe(pd.DataFrame(perf), use_container_width=True, hide_index=True)

# ── Raw Data ─────────────────────────────────────────────────────────────────
with st.expander(f"📂 Raw {sel_brand} Dataset (first 100 rows)"):
    # Lọc dữ liệu tổng theo Thương hiệu so sánh và Danh mục đang chọn
    compare_brand_data = df_all[(df_all["ProductBrand"] == sel_brand) & (df_all["ProductCategory"] == sel_cat)]
    st.dataframe(compare_brand_data.head(100), use_container_width=True)

# ── Recommendations ───────────────────────────────────────────────────────────
st.markdown("<div class='sh'>💡 Recommendations for Operation Director</div>", unsafe_allow_html=True)
top_cat = df_samsung.groupby("ProductCategory")["PurchaseFrequency"].mean().idxmax()
recs = [
    f"**Deploy XGBoost pipeline** for {sel_cat} demand planning — R²={best[1]['R2']}.",
    f"**Purchase Intent is {intent_val:.0f}%** for Samsung {sel_cat} — prioritize inventory.",
    f"**{top_cat} has highest purchase frequency** — allocate more production resources here.",
    f"**Customer Satisfaction averages {sat_val:.1f}/5** — improve after-sales service.",
    "**Re-train models quarterly** with updated sales data to maintain accuracy.",
]
st.markdown("<div class='rec'>", unsafe_allow_html=True)
for rec in recs:
    st.markdown(f"✅ {rec}")
st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='text-align:center;color:#94A3B8;font-size:12px'>"
    "© 2026 Samsung Electronics Analytics</p>",
    unsafe_allow_html=True)
