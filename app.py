import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import xgboost as xgb

# ─── 1. CẤU HÌNH TRANG STREAMLIT ───────────────────────────────────────────────
st.set_page_config(page_title="ABC Manufacturing - Demand Forecasting", layout="wide")

st.title("🏭 ABC Manufacturing — Demand Forecasting Solution")
st.markdown("### *Hệ thống Phân tích & Dự báo Nhu cầu Sản xuất dành cho Giám đốc Vận hành*")
st.divider()

# ─── 2. CẤU HÌNH GIAO DIỆN BIỂU ĐỒ ──────────────────────────────────────────────
PALETTE   = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]
BRAND_CLR = "#1E3A5F"
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
})

# ─── 3. THANH ĐIỀU HƯỚNG SIDEBAR & TẢI FILE EXCEL ──────────────────────────────
st.sidebar.header("📁 Cấu hình Dữ liệu Đầu vào")
uploaded_file = st.sidebar.file_uploader("Tải file dữ liệu Excel (.xlsx)", type=["xlsx"])

# Hàm đọc và tích hợp dữ liệu sử dụng Cache chống chậm
@st.cache_data
def load_and_clean_data(file_bytes):
    df_raw = pd.read_excel(file_bytes, sheet_name='SalesData')
    market_index = pd.read_excel(file_bytes, sheet_name='MarketIndex')
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    market_index['date'] = pd.to_datetime(market_index['date'])
    
    # Merge dữ liệu thị trường
    df_merged = df_raw.merge(market_index, on="date", how="left")
    df_cleaned = df_merged.copy()
    
    # Xử lý missing value bằng giá trị trung vị
    df_cleaned["revenue_usd"] = df_cleaned.groupby(["product", df_cleaned["date"].dt.month])["revenue_usd"].transform(lambda x: x.fillna(x.median()))
    df_cleaned["price_usd"] = (df_cleaned["revenue_usd"] / df_cleaned["units_sold"]).round(2)
    
    # Trích xuất các đặc trưng thời gian
    df_cleaned["year"] = df_cleaned["date"].dt.year
    df_cleaned["month"] = df_cleaned["date"].dt.month
    df_cleaned["quarter"] = df_cleaned["date"].dt.quarter
    
    # Định dạng Outliers (IQR)
    def flag_outlier(series):
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return ~series.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    
    df_cleaned["is_outlier"] = df_cleaned.groupby("product")["units_sold"].transform(flag_outlier)
    return df_cleaned

if uploaded_file is not None:
    try:
        # Đọc dữ liệu nhanh thông qua Cache giải phóng bộ nhớ
        df = load_and_clean_data(uploaded_file)
        st.sidebar.success("Tải file dữ liệu thành công!")
    except Exception as e:
        st.sidebar.error(f"Lỗi cấu trúc file Excel: {e}")
        st.stop()
else:
    st.info("💡 Vui lòng tải file Excel lên thanh Sidebar để khởi chạy các biểu đồ phân tích và mô hình học máy.")
    st.stop()

# Bộ lọc động tương tác dựa trên dữ liệu thực tế từ file Excel
REGIONS = df['region'].unique().tolist()
PRODUCTS = df['product'].unique().tolist()

st.sidebar.header("⚙️ Tham số Dự báo Chi tiết (Tab 2)")
selected_product = st.sidebar.selectbox("Chọn Sản phẩm Dự báo (Target Product)", PRODUCTS, index=0)
selected_region = st.sidebar.selectbox("Chọn Khu vực Dự báo (Target Region)", REGIONS, index=0)

# Hàm huấn luyện và tính toán mô hình nhanh (Được Cache lại theo điều kiện lọc)
@st.cache_data
def train_and_forecast(df_filtered, selected_product, selected_region):
    target_series = df_filtered[(df_filtered["product"] == selected_product) & (df_filtered["region"] == selected_region)].set_index("date")["units_sold"].sort_index()
    
    if len(target_series) < 15:
        return None, None, None, None
        
    def build_features(series, lags=3):
        feat = pd.DataFrame({"y": series})
        for lag in range(1, lags + 1):
            feat[f"lag_{lag}"] = feat["y"].shift(lag)
        feat["rolling_3m_mean"] = feat["y"].shift(1).rolling(3).mean()
        feat["rolling_3m_std"]  = feat["y"].shift(1).rolling(3).std()
        feat["month"]   = series.index.month
        feat["quarter"] = series.index.quarter
        feat["trend"]   = np.arange(len(feat))
        return feat.dropna()

    feat_df = build_features(target_series)
    SPLIT = int(len(feat_df) * 0.80)

    X_train, y_train = feat_df.iloc[:SPLIT].drop("y", axis=1), feat_df.iloc[:SPLIT]["y"]
    X_test, y_test = feat_df.iloc[SPLIT:].drop("y", axis=1), feat_df.iloc[SPLIT:]["y"]

    results = {}

    # --- Baseline Moving Average ---
    rolling_pred = target_series.shift(1).rolling(3).mean().reindex(y_test.index)
    results["Baseline (MA-3)"] = {
        "preds": rolling_pred.values,
        "index": rolling_pred.index,
        "MAE": float(mean_absolute_error(y_test, rolling_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, rolling_pred))),
        "R2": float(r2_score(y_test, rolling_pred)),
    }

    # --- SARIMA (Siêu tối ưu tốc độ) ---
    has_sarima = False
    sarima_forecast_vals = []
    try:
        train_ts = target_series.iloc[:SPLIT + len(feat_df) - len(target_series) + 3]
        sarima_model = SARIMAX(train_ts, order=(1, 1, 0), seasonal_order=(1, 1, 0, 12),
                               enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, method='powell', maxiter=15)
        sarima_pred = sarima_model.predict(start=y_test.index[0], end=y_test.index[-1])
        sarima_forecast_vals = sarima_model.forecast(steps=3).values
        results["SARIMA"] = {
            "preds": sarima_pred.values,
            "index": sarima_pred.index,
            "MAE": float(mean_absolute_error(y_test, sarima_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, sarima_pred))),
            "R2": float(r2_score(y_test, sarima_pred)),
        }
        has_sarima = True
    except:
        pass

    # --- Học máy XGBoost ---
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.07, max_depth=3, subsample=0.8, random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train, verbose=False)
    xgb_pred = pd.Series(xgb_model.predict(X_test), index=y_test.index)
    results["XGBoost"] = {
        "preds": xgb_pred.values,
        "index": xgb_pred.index,
        "MAE": float(mean_absolute_error(y_test, xgb_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, xgb_pred))),
        "R2": float(r2_score(y_test, xgb_pred)),
    }

    # Tính toán dự báo tương lai 3 tháng tiếp theo
    FORECAST_HORIZON = 3
    forecast_dates = pd.date_range(start=target_series.index[-1] + pd.DateOffset(months=1), periods=FORECAST_HORIZON, freq="MS")
    
    xgb_history = list(target_series.values)
    xgb_forecasts = []
    for step in range(FORECAST_HORIZON):
        lag1, lag2, lag3 = xgb_history[-1], xgb_history[-2], xgb_history[-3]
        roll_mean, roll_std = np.mean(xgb_history[-3:]), np.std(xgb_history[-3:])
        month_val = (target_series.index[-1].month + step) % 12 + 1
        qtr_val = (month_val - 1) // 3 + 1
        trend_val = len(xgb_history) + step
        row = pd.DataFrame([[lag1, lag2, lag3, roll_mean, roll_std, month_val, qtr_val, trend_val]], columns=X_train.columns)
        pred = float(xgb_model.predict(row)[0])
        xgb_forecasts.append(pred)
        xgb_history.append(pred)

    return target_series, results, forecast_dates, xgb_forecasts, sarima_forecast_vals, has_sarima

# ─── 5. GIAO DIỆN CÁC PHÂN HỆ CHỨC NĂNG (Tabs) ──────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Tổng quan & Phân tích (EDA)", "🔮 Huấn luyện & Dự báo Mô hình", "📋 Báo cáo Điều hành Toàn diện"])

# ── TAB 1: PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA) ─────────────────────────────────────
with tab1:
    st.subheader("📊 Phân tích Khám phá Dữ liệu (EDA Dashboard)")
    n_outliers = df["is_outlier"].sum()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng sản lượng đã bán toàn hệ thống", f"{df['units_sold'].sum():,}")
    col2.metric("Tổng doanh thu toàn hệ thống (USD)", f"${df['revenue_usd'].sum():,.2f}")
    col3.metric("Số dòng ngoại lai bị gán cờ", f"{n_outliers} rows ({n_outliers/len(df)*100:.1f}%)")

    # Xây dựng các tập dữ liệu tổng hợp phục vụ trực quan hóa
    monthly_total = df.groupby("date")["units_sold"].sum().reset_index()
    monthly_product = df.groupby(["date", "product"])["units_sold"].sum().unstack("product")
    monthly_region = df.groupby(["date", "region"])["units_sold"].sum().unstack("region")
    decomp = seasonal_decompose(monthly_total.set_index("date")["units_sold"], model="additive", period=12, extrapolate_trend="freq")

    fig1, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0, 0].plot(monthly_total["date"], monthly_total["units_sold"], color=PALETTE[0], linewidth=2)
    axes[0, 0].fill_between(monthly_total["date"], monthly_total["units_sold"], alpha=0.12, color=PALETTE[0])
    axes[0, 0].set_title("Tổng Nhu cầu Sản xuất Hàng tháng")

    for i, col in enumerate(monthly_product.columns):
        axes[0, 1].plot(monthly_product.index, monthly_product[col], label=col, color=PALETTE[i], linewidth=1.8)
    axes[0, 1].set_title("Nhu cầu Theo Từng Dòng Sản phẩm")
    axes[0, 1].legend(fontsize=9)

    axes[1, 0].stackplot(monthly_region.index, [monthly_region[r] for r in REGIONS], labels=REGIONS, colors=PALETTE[:4], alpha=0.75)
    axes[1, 0].set_title("Tỷ trọng Phân bổ Nhu cầu theo Khu vực")
    axes[1, 0].legend(loc="upper left", fontsize=9)

    axes[1, 1].bar(decomp.seasonal.index, decomp.seasonal.values, color=PALETTE[2], alpha=0.8, width=15)
    axes[1, 1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1, 1].set_title("Thành phần Mùa vụ (Additive Decomposition)")

    plt.tight_layout()
    st.pyplot(fig1)

# ── TAB 2: HUẤN LUYỆN MÔ HÌNH MACHINE LEARNING & ĐÁNH GIÁ ───────────────────────
with tab2:
    st.subheader(f"🔮 Dự báo Nhu cầu cho: {selected_product} tại Khu vực: {selected_region}")
    
    # Gọi hàm dự báo từ Cache (Chạy siêu tốc)
    target_series, results, forecast_dates, xgb_forecasts, sarima_forecast_vals, has_sarima = train_and_forecast(df, selected_product, selected_region)

    if target_series is None:
        st.error("Dữ liệu chuỗi thời gian quá ngắn.")
    else:
        # Hiển thị chỉ số đánh giá mô hình
        eval_data = {name: {"MAE": v["MAE"], "RMSE": v["RMSE"], "R²": v["R2"]} for name, v in results.items()}
        st.dataframe(pd.DataFrame(eval_data).T.round(2), use_container_width=True)

        # Trực quan hóa kết quả đồ thị
        fig2 = plt.figure(figsize=(14, 5))
        plt.plot(target_series.index, target_series.values, "o-", color="black", linewidth=2, label="Dữ liệu Thực tế (Actual)", zorder=5)
        
        for idx, (name, res) in enumerate(results.items()):
            plt.plot(res["index"], res["preds"], "--", linewidth=1.5, color=PALETTE[idx], label=f"Mô phỏng {name}")
        
        plt.axvspan(forecast_dates[0], forecast_dates[-1], alpha=0.08, color=PALETTE[4], label="Cửa sổ Dự báo Tương lai")
        
        if has_sarima:
            plt.plot(forecast_dates, sarima_forecast_vals, "s--", color=PALETTE[1], markersize=6, label="Dự báo bằng SARIMA")
        plt.plot(forecast_dates, xgb_forecasts, "^--", color=PALETTE[2], markersize=6, label="Dự báo bằng XGBoost")
        
        plt.title(f"Mô hình hóa Mô phỏng & Dự báo cho {selected_product} tại Khu vực {selected_region}")
        plt.ylabel("Số lượng sản phẩm")
        plt.legend(fontsize=9, loc="upper left")
        st.pyplot(fig2)

        # Bảng hiển thị kết quả dự báo
        forecast_table_data = {
            "Tháng Tương lai": forecast_dates.strftime("%B %Y"),
            "Dự báo XGBoost (Units)": np.round(xgb_forecasts).astype(int),
        }
        if has_sarima:
            forecast_table_data["Dự báo SARIMA (Units)"] = np.round(sarima_forecast_vals).astype(int)
            
        st.table(pd.DataFrame(forecast_table_data))

# ── TAB 3: BÁO CÁO ĐIỀU HÀNH TOÀN DIỆN ──────────────────────────────────────────
with tab3:
    st.subheader("📋 Báo cáo Chiến lược Điều hành Toàn diện (Toàn bộ Hệ thống)")
    st.markdown(f"""
    * **Chu kỳ Tập dữ liệu phân tích:** {df['date'].min().strftime('%m/%Y')} ➔ {df['date'].max().strftime('%m/%Y')}
    * **Tổng Sản lượng Toàn hệ thống:** **{df['units_sold'].sum():,}** sản phẩm.
    * **Tổng Doanh thu Tích lũy toàn hệ thống:** **${df['revenue_usd'].sum():,.2f} USD**.
    """)
    st.divider()
    
    st.markdown("### 🗺️ 1. Số liệu Thống kê Đầy đủ theo Từng Khu vực (All Regions Summary)")
    region_summary = df.groupby("region").agg(Tong_San_Luong=("units_sold", "sum"), Doanh_Thu_USD=("revenue_usd", "sum"), Gia_Binh_Quan=("price_usd", "mean")).round(2)
    region_summary.columns = ["Tổng Sản Lượng (Units)", "Tổng Doanh Thu (USD)", "Giá Bán Bình Quân (USD/Unit)"]
    st.dataframe(region_summary, use_container_width=True)
    
    st.markdown("### 📦 2. Số liệu Thống kê Đầy đủ theo Từng Dòng Sản phẩm (All Products Summary)")
    product_summary = df.groupby("product").agg(Tong_San_Luong=("units_sold", "sum"), Doanh_Thu_USD=("revenue_usd", "sum"), Gia_Binh_Quan=("price_usd", "mean")).round(2)
    product_summary.columns = ["Tổng Sản Lượng (Units)", "Tổng Doanh Thu (USD)", "Giá Bán Bình Quân (USD/Unit)"]
    st.dataframe(product_summary, use_container_width=True)

    st.markdown("### 📈 3. Ma Trận Sản Lượng Phân Phối Chi tiết (Vùng miền × Sản phẩm)")
    st.dataframe(df.pivot_table(values="units_sold", index="region", columns="product", aggfunc="sum"), use_container_width=True)
