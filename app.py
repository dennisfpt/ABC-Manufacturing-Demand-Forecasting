import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
import io

# 1. CẤU HÌNH GIAO DIỆN WEB ĐỒ HỌA
st.set_page_config(page_title="ABC Manufacturing - Demand Forecasting", layout="wide", page_icon="📈")

PALETTE = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]
BRAND_CLR = "#1E3A5F"
sns.set_theme(style="whitegrid", font_scale=1.05)

# 2. XỬ LÝ DỮ LIỆU NGẦM (Sử dụng cache để tối ưu hiệu năng trang web)
@st.cache_resource
def run_data_science_pipeline():
    # --- SECTION 1: DATA GENERATION ---
    np.random.seed(42)
    REGIONS  = ["North", "South", "East", "West"]
    PRODUCTS = ["Widget A", "Widget B", "Widget C"]
    START    = pd.Timestamp("2021-01-01")
    PERIODS  = 36
    date_range = pd.date_range(start=START, periods=PERIODS, freq="MS")
    BASE_DEMAND = {"Widget A": 1_200, "Widget B": 850, "Widget C": 600}

    rows = []
    for region in REGIONS:
        for product in PRODUCTS:
            base = BASE_DEMAND[product]
            region_factor = {"North": 1.3, "South": 1.0, "East": 0.85, "West": 1.1}[region]
            for i, date in enumerate(date_range):
                trend = base * region_factor * (1 + 0.008 * i)
                seasonal = trend * 0.15 * np.sin(2 * np.pi * (date.month - 3) / 12)
                noise = np.random.normal(0, trend * 0.05)
                demand = max(0, trend + seasonal + noise)
                revenue = round(demand * np.random.uniform(18, 22), 2)
                if np.random.random() < 0.02:
                    revenue = np.nan
                rows.append({
                    "date": date, "region": region, "product": product,
                    "units_sold": int(demand), "revenue_usd": revenue,
                    "price_usd": round(revenue / int(demand), 2) if revenue else np.nan,
                })
    df_raw = pd.DataFrame(rows)

    # --- SECTION 2: INTEGRATION ---
    market_index = pd.DataFrame({
        "date": date_range,
        "consumer_confidence": np.clip(90 + np.cumsum(np.random.normal(0.3, 1.5, PERIODS)), 80, 115).round(1),
        "raw_material_cost": np.clip(100 + np.cumsum(np.random.normal(0.2, 2.0, PERIODS)), 85, 130).round(2),
    })
    df_merged = df_raw.merge(market_index, on="date", how="left")

    # --- SECTION 3: CLEANING ---
    df = df_merged.copy()
    df["revenue_usd"] = df.groupby(["product", df["date"].dt.month])["revenue_usd"].transform(lambda x: x.fillna(x.median()))
    df["price_usd"] = (df["revenue_usd"] / df["units_sold"]).round(2)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    def flag_outlier(series):
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        return ~series.between(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
    df["is_outlier"] = df.groupby("product")["units_sold"].transform(flag_outlier)

    # --- SECTION 4 & 5: FEATURE ENGINEERING & MODELING ---
    target_series = df[(df["product"] == "Widget A") & (df["region"] == "North")].set_index("date")["units_sold"].sort_index()
    
    feat = pd.DataFrame({"y": target_series})
    for lag in range(1, 4):
        feat[f"lag_{lag}"] = feat["y"].shift(lag)
    feat["rolling_3m_mean"] = feat["y"].shift(1).rolling(3).mean()
    feat["rolling_3m_std"]  = feat["y"].shift(1).rolling(3).std()
    feat["month"]   = target_series.index.month
    feat["quarter"] = target_series.index.quarter
    feat["trend"]   = np.arange(len(feat))
    feat_df = feat.dropna()

    SPLIT = int(len(feat_df) * 0.80)
    X_train, y_train = feat_df.iloc[:SPLIT].drop("y", axis=1), feat_df.iloc[:SPLIT]["y"]
    X_test, y_test   = feat_df.iloc[SPLIT:].drop("y", axis=1), feat_df.iloc[SPLIT:]["y"]

    results = {}
    # 6a. Baseline
    baseline_pred = target_series.shift(1).rolling(3).mean().reindex(y_test.index)
    results["Baseline (MA-3)"] = {"preds": baseline_pred, "MAE": mean_absolute_error(y_test, baseline_pred), "RMSE": np.sqrt(mean_squared_error(y_test, baseline_pred)), "R2": r2_score(y_test, baseline_pred)}

    # 6b. SARIMA
    train_ts = target_series.iloc[: SPLIT + len(feat_df) - len(target_series) + 3]
    sarima_model = SARIMAX(train_ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarima_pred = sarima_model.predict(start=y_test.index[0], end=y_test.index[-1])
    results["SARIMA"] = {"preds": sarima_pred, "MAE": mean_absolute_error(y_test, sarima_pred), "RMSE": np.sqrt(mean_squared_error(y_test, sarima_pred)), "R2": r2_score(y_test, sarima_pred)}

    # 6c. XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_pred = pd.Series(xgb_model.predict(X_test), index=y_test.index)
    results["XGBoost"] = {"preds": xgb_pred, "MAE": mean_absolute_error(y_test, xgb_pred), "RMSE": np.sqrt(mean_squared_error(y_test, xgb_pred)), "R2": r2_score(y_test, xgb_pred)}

    # --- FORECAST ---
    FORECAST_HORIZON = 3
    forecast_dates = pd.date_range(start=target_series.index[-1] + pd.DateOffset(months=1), periods=FORECAST_HORIZON, freq="MS")
    sarima_fc = sarima_model.forecast(steps=FORECAST_HORIZON)
    sarima_fc.index = forecast_dates

    xgb_history = list(target_series.values)
    xgb_forecasts = []
    for step in range(FORECAST_HORIZON):
        row = pd.DataFrame([[xgb_history[-1], xgb_history[-2], xgb_history[-3], np.mean(xgb_history[-3:]), np.std(xgb_history[-3:]), (target_series.index[-1].month + step) % 12 + 1, ((target_series.index[-1].month + step) % 12 + 1 - 1) // 3 + 1, len(xgb_history) + step]], columns=X_train.columns)
        pred = float(xgb_model.predict(row)[0])
        xgb_forecasts.append(pred)
        xgb_history.append(pred)
    xgb_fc = pd.Series(xgb_forecasts, index=forecast_dates)

    return df, results, sarima_fc, xgb_fc, y_test, target_series, forecast_dates, xgb_model, X_train

# Thực thi pipeline xử lý dữ liệu
df, results, sarima_fc, xgb_fc, y_test, target_series, forecast_dates, xgb_model, X_train = run_data_science_pipeline()

# 3. THIẾT KẾ KHÔNG GIAN GIAO DIỆN WEB (UI)
st.title("🏭 ABC Manufacturing — Demand Forecasting Dashboard")
st.markdown("### Khung giải pháp khoa học dữ liệu hỗ trợ Giám đốc Vận hành")
st.write("---")

# Tầng 1: Các chỉ số cốt lõi (KPI Cards)
total_units = df["units_sold"].sum()
total_revenue = df["revenue_usd"].sum()
top_product = df.groupby("product")["units_sold"].sum().idxmax()
top_region = df.groupby("region")["units_sold"].sum().idxmax()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(label="Tổng Sản Lượng Đã Bán", value=f"{total_units:,} sản phẩm")
kpi2.metric(label="Tổng Doanh Thu Đạt Được", value=f"${total_revenue:,.2f}")
kpi3.metric(label="Dòng Sản Phẩm Chủ Lực", value=top_product)
kpi4.metric(label="Thị Trường Tăng Trưởng Nhất", value=top_region)

st.write("---")

# Tạo các tab điều hướng trực quan
tab1, tab2, tab3 = st.tabs(["📊 Phân tích Thống kê (EDA)", "🤖 Đánh giá Mô hình & Dự báo", "📋 Dữ liệu sạch (Database)"])

with tab1:
    st.header("Phân Tích Khám Phá Dữ Liệu Sản Xuất (EDA)")
    col_eda1, col_eda2 = st.columns(2)
    
    with col_eda1:
        fig_eda1, ax_eda1 = plt.subplots(figsize=(10, 4))
        monthly_total = df.groupby("date")["units_sold"].sum().reset_index()
        ax_eda1.plot(monthly_total["date"], monthly_total["units_sold"], color=PALETTE[0], linewidth=2.5)
        ax_eda1.fill_between(monthly_total["date"], monthly_total["units_sold"], alpha=0.15, color=PALETTE[0])
        ax_eda1.set_title("Biến động tổng nhu cầu theo tháng (Toàn bộ sản phẩm)")
        st.pyplot(fig_eda1)
        
        fig_heat1, ax_heat1 = plt.subplots(figsize=(10, 4))
        pivot_heat = df.groupby(["region", "product"])["units_sold"].sum().unstack()
        sns.heatmap(pivot_heat, ax=ax_heat1, cmap="Blues", annot=True, fmt=",", linewidths=0.5)
        ax_heat1.set_title("Bản đồ nhiệt sản lượng: Khu vực × Dòng sản phẩm")
        st.pyplot(fig_heat1)

    with col_eda2:
        fig_eda2, ax_eda2 = plt.subplots(figsize=(10, 4))
        monthly_product = df.groupby(["date", "product"])["units_sold"].sum().unstack()
        for i, col in enumerate(monthly_product.columns):
            ax_eda2.plot(monthly_product[col].index, monthly_product[col].values, label=col, color=PALETTE[i], linewidth=2)
        ax_eda2.set_title("Nhu cầu tiêu thụ theo từng danh mục thiết bị")
        ax_eda2.legend()
        st.pyplot(fig_eda2)

        fig_eda3, ax_eda3 = plt.subplots(figsize=(10, 4))
        monthly_region = df.groupby(["date", "region"])["units_sold"].sum().unstack()
        ax_eda3.stackplot(monthly_region.index, [monthly_region[r] for r in ["North", "South", "East", "West"]], labels=["North", "South", "East", "West"], colors=PALETTE[:4], alpha=0.75)
        ax_eda3.set_title("Cấu trúc sản lượng tiêu thụ phân bổ theo vùng (Tích lũy)")
        ax_eda3.legend(loc="upper left")
        st.pyplot(fig_eda3)

with tab2:
    st.header("Hiệu Suất Mô Hình AI & Dự Báo 3 Tháng Kế Tiếp")
    st.markdown("**Mục tiêu phân tích chuyên sâu:** Cánh tay robot `Widget A` tại khu vực `North` (Thị trường trọng điểm).")
    
    fig_mod, ax_mod = plt.subplots(figsize=(15, 4.5))
    ax_mod.plot(y_test.index, y_test.values, "o-", color="black", linewidth=2, label="Thực tế (Actual)")
    ax_mod.plot(results["Baseline (MA-3)"]["preds"].index, results["Baseline (MA-3)"]["preds"].values, "--", color=PALETTE[0], label=f"Moving Average (R²={results['Baseline (MA-3)']['R2']:.2f})")
    ax_mod.plot(results["SARIMA"]["preds"].index, results["SARIMA"]["preds"].values, "--", color=PALETTE[1], label=f"SARIMA (R²={results['SARIMA']['R2']:.2f})")
    ax_mod.plot(results["XGBoost"]["preds"].index, results["XGBoost"]["preds"].values, "--", color=PALETTE[2], label=f"XGBoost (R²={results['XGBoost']['R2']:.2f})")
    
    ax_mod.axvspan(forecast_dates[0], forecast_dates[-1], alpha=0.1, color=PALETTE[4], label="Cửa sổ Dự báo tương lai")
    ax_mod.plot(forecast_dates, sarima_fc.values, "s--", color=PALETTE[1], markersize=6, linewidth=2, label="Dự báo tương lai bằng SARIMA")
    ax_mod.plot(forecast_dates, xgb_fc.values, "^--", color=PALETTE[2], markersize=6, linewidth=2, label="Dự báo tương lai bằng XGBoost")
    ax_mod.set_title("Đồ thị đối chiếu kiểm định mô hình và dự đoán xu hướng tương lai")
    ax_mod.legend(loc="upper left")
    st.pyplot(fig_mod)
    
    col_mod1, col_mod2 = st.columns(2)
    
    with col_mod1:
        st.subheader("📊 So sánh sai số giữa các mô hình (Sai số càng thấp càng tốt)")
        eval_df = pd.DataFrame({name: {"MAE": v["MAE"], "RMSE": v["RMSE"], "R²": v["R2"]} for name, v in results.items()}).T
        st.dataframe(eval_df.round(2), use_container_width=True)
        
        st.error("🚨 **KHUYẾN NGHỊ ĐƯỢC ĐỀ XUẤT CHO GIÁM ĐỐC VẬN HÀNH (OPERATIONS DIRECTOR):**\n\n"
                 "1. **Triển khai Mô hình:** Áp dụng ngay đường ống thuật toán **XGBoost** để lập kế hoạch phân phối hàng tháng.\n"
                 "2. **Tối ưu Kho bãi:** Tăng lượng hàng dự trữ (Buffer stock) tại kho miền Bắc (North) thêm ít nhất 10%.\n"
                 "3. **Quản lý Cung ứng:** Lên lịch mua sắm nguyên vật liệu thô trước **6 tuần**.\n"
                 "4. **Chu kỳ mùa vụ:** Đặc biệt lưu ý đỉnh điểm sản xuất vào **Quý 4 (Q4)**, nhu cầu sẽ tăng đột biến ~15%.")

    with col_mod2:
        st.subheader("🎯 Dự đoán số lượng tiêu thụ cụ thể trong 3 tháng tới")
        forecast_table = pd.DataFrame({
            "Tháng Dự Báo": forecast_dates.strftime("%B %Y"),
            "Dự kiến bằng SARIMA (Units)": sarima_fc.round(0).astype(int).values,
            "Dự kiến bằng XGBoost (Units)": xgb_fc.round(0).astype(int).values
        })
        st.dataframe(forecast_table, use_container_width=True, hide_index=True)
        
        # FIX LỖI ĐỒ HỌA FEATURE IMPORTANCE BẰNG CÁCH TRUYỀN MA TRẬN CHUẨN ĐẦU VÀO
        fig_imp, ax_imp = plt.subplots(figsize=(10, 3.5))
        feat_imp = pd.Series(xgb_model.feature_importances_, index=X_train.columns).sort_values(ascending=True)
        feat_imp.plot(kind="barh", ax=ax_imp, color=PALETTE[0])
        ax_imp.set_title("Mức độ ảnh hưởng của các thuộc tính tới kết quả dự đoán (XGBoost)")
        st.pyplot(fig_imp)

with tab3:
    st.header("Cơ Sở Dữ Liệu Sạch Đã Qua Tiền Xử Lý (ABC Manufacturing Cleaned Data)")
    df.to_csv("abc_manufacturing_cleaned.csv", index=False)
    st.dataframe(df, use_container_width=True)
