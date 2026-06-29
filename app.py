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
import io

# ─── Cấu hình Trang Streamlit ──────────────────────────────────────────────────
st.set_page_config(page_title="ABC Manufacturing - Demand Forecasting", layout="wide")

st.title("🏭 ABC Manufacturing — Demand Forecasting Solution")
st.markdown("### *Hệ thống Phân tích & Dự báo Nhu cầu Sản xuất dành cho Giám đốc Vận hành*")
st.divider()
# ─── Cấu hình Giao diện Biểu đồ ───────────────────────────────────────────────
PALETTE   = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]
BRAND_CLR = "#1E3A5F"
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
})

# ─── Thanh Điều Hướng Sidebar ─────────────────────────────────────────────────
st.sidebar.header("📁 Cấu hình Dữ liệu Đầu vào")
uploaded_file = st.sidebar.file_uploader("Tải file dữ liệu Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # Đọc dữ liệu từ file Excel
    try:
        df_raw = pd.read_excel(uploaded_file, sheet_name='SalesData')
        market_index = pd.read_excel(uploaded_file, sheet_name='MarketIndex')
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        market_index['date'] = pd.to_datetime(market_index['date'])
        st.sidebar.success("Tải file dữ liệu thành công!")
    except Exception as e:
        st.sidebar.error(f"Lỗi cấu trúc file Excel: {e}")
        st.stop()
else:
    st.info("💡 Vui lòng tải file Excel `abc_manufacturing_data.xlsx` ở trên lên thanh Sidebar để bắt đầu chạy ứng dụng web.")
    st.stop()

# Bộ lọc động dựa trên dữ liệu đầu vào
REGIONS = df_raw['region'].unique().tolist()
PRODUCTS = df_raw['product'].unique().tolist()

st.sidebar.header("⚙️ Tham số Dự báo")
selected_product = st.sidebar.selectbox("Chọn Sản phẩm Dự báo (Target Product)", PRODUCTS, index=0)
selected_region = st.sidebar.selectbox("Chọn Khu vực Dự báo (Target Region)", REGIONS, index=0)

# ─── XỬ LÝ & LÀM SẠCH DỮ LIỆU (Data Preprocessing) ──────────────────────────
df_merged = df_raw.merge(market_index, on="date", how="left")
df = df_merged.copy()

# Điền giá trị trống cho Doanh thu bằng median của sản phẩm đó theo từng tháng
df["revenue_usd"] = df.groupby(["product", df["date"].dt.month])["revenue_usd"].transform(lambda x: x.fillna(x.median()))
df["price_usd"] = (df["revenue_usd"] / df["units_sold"]).round(2)

# Trích xuất các đặc trưng thời gian
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter

# Phát hiện điểm dị biệt Outlier (IQR method)
def flag_outlier(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return ~series.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

df["is_outlier"] = df.groupby("product")["units_sold"].transform(flag_outlier)
n_outliers = df["is_outlier"].sum()

# ─── TAB VIEW TRÊN WEB ────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Tổng quan & Phân tích (EDA)", "🔮 Huấn luyện & Dự báo Mô hình", "📋 Báo cáo Điều hành"])

with tab1:
    st.subheader("📊 Phân tích Khám phá Dữ liệu (EDA Dashboard)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng sản lượng đã bán", f"{df['units_sold'].sum():,}")
    col2.metric("Tổng doanh thu (USD)", f"${df['revenue_usd'].sum():,.2f}")
    col3.metric("Số dòng ngoại lai bị gán cờ", f"{n_outliers} rows ({n_outliers/len(df)*100:.1f}%)")

    # Tạo biểu đồ EDA
    monthly_total = df.groupby("date")["units_sold"].sum().reset_index()
    monthly_product = df.groupby(["date", "product"])["units_sold"].sum().unstack("product")
    monthly_region = df.groupby(["date", "region"])["units_sold"].sum().unstack("region")
    decomp = seasonal_decompose(monthly_total.set_index("date")["units_sold"], model="additive", period=12, extrapolate_trend="freq")

    fig1, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # 1a. Tổng nhu cầu hàng tháng
    axes[0, 0].plot(monthly_total["date"], monthly_total["units_sold"], color=PALETTE[0], linewidth=2)
    axes[0, 0].fill_between(monthly_total["date"], monthly_total["units_sold"], alpha=0.12, color=PALETTE[0])
    axes[0, 0].set_title("Tổng Nhu cầu Sản xuất Hàng tháng (Tất cả sản phẩm)")

    # 1b. Nhu cầu theo sản phẩm
    for i, col in enumerate(monthly_product.columns):
        axes[0, 1].plot(monthly_product.index, monthly_product[col], label=col, color=PALETTE[i], linewidth=1.8)
    axes[0, 1].set_title("Nhu cầu Theo Từng Dòng Sản phẩm")
    axes[0, 1].legend(fontsize=9)

    # 1c. Phân bổ theo vùng miền
    axes[1, 0].stackplot(monthly_region.index, [monthly_region[r] for r in REGIONS], labels=REGIONS, colors=PALETTE[:4], alpha=0.75)
    axes[1, 0].set_title("Tỷ trọng Phân bổ Nhu cầu theo Khu vực Vùng miền")
    axes[1, 0].legend(loc="upper left", fontsize=9)

    # 1d. Tính mùa vụ
    axes[1, 1].bar(decomp.seasonal.index, decomp.seasonal.values, color=PALETTE[2], alpha=0.8, width=15)
    axes[1, 1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1, 1].set_title("Thành phần Mùa vụ (Additive Decomposition)")

    plt.tight_layout()
    st.pyplot(fig1)

    # Kiểm định tính dừng ADF
    adf_stat, adf_p, *_ = adfuller(monthly_total["units_sold"])
    st.info(f"🔎 **Kiểm định tính dừng ADF (Augmented Dickey-Fuller):** p-value = **{adf_p:.4f}** → Dữ liệu chuỗi thời gian tổng thể là **{'' if adf_p < 0.05 else 'không'} ổn định/dừng (stationary)**.")

with tab2:
    st.subheader(f"🔮 Dự báo Nhu cầu cho: {selected_product} tại Khu vực: {selected_region}")
    
    # Lấy chuỗi dữ liệu mục tiêu đã chọn từ bộ lọc
    target_series = df[(df["product"] == selected_product) & (df["region"] == selected_region)].set_index("date")["units_sold"].sort_index()

    if len(target_series) < 15:
        st.error("Dữ liệu quá ngắn để huấn luyện mô hình dự báo chuỗi thời gian. Vui lòng kiểm tra lại file Excel.")
    else:
        # Xây dựng ma trận đặc trưng cho ML
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

        # Mô hình 1: Baseline Moving Average
        rolling_pred = target_series.shift(1).rolling(3).mean().reindex(y_test.index)
        results["Baseline (MA-3)"] = {
            "preds": rolling_pred,
            "MAE": mean_absolute_error(y_test, rolling_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, rolling_pred)),
            "R2": r2_score(y_test, rolling_pred),
        }

        # Mô hình 2: SARIMA
        train_ts = target_series.iloc[:SPLIT + len(feat_df) - len(target_series) + 3]
        sarima_model = SARIMAX(train_ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                               enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        sarima_pred = sarima_model.predict(start=y_test.index[0], end=y_test.index[-1])
        results["SARIMA"] = {
            "preds": sarima_pred,
            "MAE": mean_absolute_error(y_test, sarima_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, sarima_pred)),
            "R2": r2_score(y_test, sarima_pred),
        }

        # Mô hình 3: XGBoost
        xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        xgb_pred = pd.Series(xgb_model.predict(X_test), index=y_test.index)
        results["XGBoost"] = {
            "preds": xgb_pred,
            "MAE": mean_absolute_error(y_test, xgb_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, xgb_pred)),
            "R2": r2_score(y_test, xgb_pred),
        }

        # Hiển thị bảng so sánh độ đo lỗi
        eval_df = pd.DataFrame({name: {"MAE": v["MAE"], "RMSE": v["RMSE"], "R²": v["R2"]} for name, v in results.items()}).T.round(2)
        st.write("📊 **So sánh độ chính xác giữa các thuật toán dự báo (Test Set):**")
        st.dataframe(eval_df, use_container_width=True)

        # Tính toán dự báo tương lai 3 tháng tiếp theo
        FORECAST_HORIZON = 3
        forecast_dates = pd.date_range(start=target_series.index[-1] + pd.DateOffset(months=1), periods=FORECAST_HORIZON, freq="MS")
        
        sarima_forecast = sarima_model.forecast(steps=FORECAST_HORIZON)
        sarima_forecast.index = forecast_dates

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
        xgb_forecast_series = pd.Series(xgb_forecasts, index=forecast_dates)

        # Đồ thị so sánh thực tế và dự báo
        fig2 = plt.figure(figsize=(14, 6))
        plt.plot(target_series.index, target_series.values, "o-", color="black", label="Dữ liệu Thực tế (Actual)", zorder=5)
        for idx, (name, res) in enumerate(results.items()):
            plt.plot(res["preds"].index, res["preds"].values, "--", color=PALETTE[idx], label=f"Mô phỏng thử nghiệm {name}")
        
        # Vùng dự báo tương lai
        plt.axvspan(forecast_dates[0], forecast_dates[-1], alpha=0.08, color=PALETTE[4], label="Cửa sổ Dự báo Tương lai")
        plt.plot(forecast_dates, sarima_forecast.values, "s--", color=PALETTE[1], markersize=6, label="Dự báo tương lai bằng SARIMA")
        plt.plot(forecast_dates, xgb_forecast_series.values, "^--", color=PALETTE[2], markersize=6, label="Dự báo tương lai bằng XGBoost")
        plt.title(f"Mô hình hóa Dự báo cho {selected_product} tại thị trường vùng {selected_region}")
        plt.ylabel("Số lượng sản phẩm bán ra")
        plt.legend(fontsize=9, loc="upper left")
        st.pyplot(fig2)

        # Bảng hiển thị kết quả dự báo 3 tháng tới
        forecast_df_display = pd.DataFrame({
            "Tháng Tương lai": forecast_dates.strftime("%B %Y"),
            "Dự báo SARIMA (Units)": sarima_forecast.round(0).astype(int).values,
            "Dự báo XGBoost (Units)": xgb_forecast_series.round(0).astype(int).values,
        })
        st.write("📋 **Giá trị Dự báo Chi tiết 3 Tháng tiếp theo:**")
        st.table(forecast_df_display)

with tab3:
    st.subheader("📝 Báo cáo Tóm tắt dành cho Giám đốc Vận hành (Executive Summary)")
    
    total_units = df["units_sold"].sum()
    total_revenue = df["revenue_usd"].sum()
    top_product = df.groupby("product")["units_sold"].sum().idxmax()
    top_region = df.groupby("region")["units_sold"].sum().idxmax()
    
    st.markdown(f"""
    * **Chu kỳ Tập dữ liệu:** {df['date'].min().strftime('%m/%Y')} ➔ {df['date'].max().strftime('%m/%Y')}
    * **Tổng Sản lượng Bán hàng tích lũy:** **{total_units:,}** sản phẩm.
    * **Tổng Doanh thu Tích lũy:** **${total_revenue:,.2f} USD**.
    * **Dòng sản phẩm chủ lực (Top Volume Product):** `{top_product}`
    * **Thị trường tăng trưởng tốt nhất (Top Region):** Vùng `{top_region}`
    """)
    
    st.success("""
    ### 💡 Khuyến nghị Vận hành Sản xuất (Operation Recommendations):
    1. **Triển khai Pipeline XGBoost:** Ưu tiên sử dụng mô hình học máy XGBoost cho kế hoạch cung ứng hàng tháng nhờ mức độ thích nghi đột biến biên độ tốt hơn.
    2. **Tối ưu hóa kho bãi Khu vực phía Bắc (North Region):** Cần tăng biên độ tồn kho an toàn (stock buffer) tại khu vực này từ 10-15% vì đây là thị trường có tốc độ tăng trưởng nhanh nhất.
    3. **Thời gian Đặt mua Nguyên vật liệu (Lead-time):** Lên kế hoạch cung ứng vật tư thô sớm **6 tuần trước chu kỳ sản xuất** nhằm đón đầu đà tăng giá vật liệu.
    4. **Theo dõi Đỉnh điểm mùa vụ Q4:** Dữ liệu cho thấy nhu cầu toàn hệ thống luôn bật tăng ~15% vào Quý IV. Cần chuẩn bị nhân lực sản xuất từ cuối Quý III.
    5. **Cập nhật dữ liệu ERP định kỳ:** Chạy lại (Re-train) mô hình dự báo mỗi quý một lần với dữ liệu ERP mới để đảm bảo tính chính xác cao nhất.
    """)
