"""
============================================================
 ABC MANUFACTURING — DEMAND FORECASTING WEB SOLUTION
 Data Science Case Study | Operation Director Support
 Author  : Junior Analyst, ABC Manufacturing
 Version : 2.0 (Streamlit Web App Version)
============================================================
"""

# ─── Imports ─────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import xgboost as xgb
import os
import datetime

# 1. CẤU HÌNH GIAO DIỆN WEB (TỰ ĐỘNG BẬT TOÀN MÀN HÌNH CHỐNG VỠ CHỮ)
st.set_page_config(
    page_title="ABC Manufacturing - Demand Forecasting", 
    layout="wide", 
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# ─── Global Style ─────────────────────────────────────────────────────────────
PALETTE   = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]
BRAND_CLR = "#1E3A5F"
BG_GRADIENT = "linear-gradient(135deg, #1E3A5F 0%, #3B82F6 100%)"

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
})

DATA_FILENAME = "ABC_Manufacturing_IoT_Simulation_Data.xlsx"

# =============================================================================
# DATA PIPELINE NGẦM (XỬ LÝ TOÀN BỘ LOGIC ML & CACHE ĐỂ TĂNG TỐC WEB)
# =============================================================================
@st.cache_resource
def run_data_science_pipeline():
    if not os.path.exists(DATA_FILENAME):
        return None
        
    # SECTION 1 & 2: Đọc dữ liệu từ file Excel thực tế
    df_excel = pd.read_excel(DATA_FILENAME)
    df = df_excel.copy()
    df["Ngày_Kiểm_Tra"] = pd.to_datetime(df["Ngày_Kiểm_Tra"])
    
    # SECTION 3: Làm sạch dữ liệu lỗi định dạng ô đặc trưng của Excel
    def fix_excel_format_direct(val, default_val):
        if pd.isna(val): return default_val
        if isinstance(val, (pd.Timestamp, datetime.date)) or hasattr(val, 'day'):
            try: return float(val.day) + float(val.month) / 10.0
            except: return default_val
        val_str = str(val).strip()
        if "-" in val_str:
            try:
                parts = val_str.split("-")
                return float(parts[2]) + float(parts[1]) / 10.0
            except: return default_val
        try: return float(val)
        except: return default_val

    df["Độ_Rung_mm_s"] = df["Độ_Rung_mm_s"].apply(lambda x: fix_excel_format_direct(x, 2.5))
    df["Điện_Năng_kWh"] = df["Điện_Năng_kWh"].apply(lambda x: fix_excel_format_direct(x, 12.0))
    df["Nhiệt_Độ_C"] = pd.to_numeric(df["Nhiệt_Độ_C"], errors='coerce').fillna(df["Nhiệt_Độ_C"].median())
    
    df["year"]    = df["Ngày_Kiểm_Tra"].dt.year
    df["month"]   = df["Ngày_Kiểm_Tra"].dt.month
    df["quarter"] = df["Ngày_Kiểm_Tra"].dt.quarter

    # Outliers detection
    q1, q3 = df["Nhiệt_Độ_C"].quantile(0.25), df["Nhiệt_Độ_C"].quantile(0.75)
    iqr = q3 - q1
    df["is_outlier"] = ~df["Nhiệt_Độ_C"].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    
    # SECTION 4: Chuẩn bị chuỗi EDA thời gian
    daily_total = df.groupby("Ngày_Kiểm_Tra")["Nhiệt_Độ_C"].mean().reset_index()
    daily_total.columns = ["date", "avg_temp"]
    decomp = seasonal_decompose(daily_total.set_index("date")["avg_temp"], model="additive", period=7, extrapolate_trend="freq")
    
    # SECTION 5: Feature Engineering cho thiết bị trọng điểm SMT-001
    target_series = df[df["Mã_Thiết_Bị"] == "SMT-001"].set_index("Ngày_Kiểm_Tra")["Nhiệt_Độ_C"].sort_index()
    
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
    X_train, X_test = feat_df.iloc[:SPLIT].drop("y", axis=1), feat_df.iloc[SPLIT:].drop("y", axis=1)
    y_train, y_test = feat_df.iloc[:SPLIT]["y"], feat_df.iloc[SPLIT:]["y"]
    
    # SECTION 6: Huấn luyện mô hình chuyên sâu (Moving Average, SARIMA, XGBoost)
    results = {}
    
    # 6a. Baseline
    baseline_pred = target_series.shift(1).rolling(3).mean().reindex(y_test.index)
    results["Baseline (MA-3)"] = {"preds": baseline_pred, "MAE": mean_absolute_error(y_test, baseline_pred), "R2": r2_score(y_test, baseline_pred)}
    
    # 6b. SARIMA
    train_ts = target_series.iloc[: SPLIT + len(feat_df) - len(target_series) + 3]
    sarima_model = SARIMAX(train_ts, order=(1,1,1), seasonal_order=(1,1,1,7), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarima_pred = sarima_model.predict(start=y_test.index[0], end=y_test.index[-1])
    results["SARIMA(1,1,1)(1,1,1,7)"] = {"preds": sarima_pred, "MAE": mean_absolute_error(y_test, sarima_pred), "R2": r2_score(y_test, sarima_pred)}
    
    # 6c. XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train)
    xgb_pred = pd.Series(xgb_model.predict(X_test), index=y_test.index)
    results["XGBoost"] = {"preds": xgb_pred, "MAE": mean_absolute_error(y_test, xgb_pred), "R2": r2_score(y_test, xgb_pred)}
    
    # SECTION 8: Dự báo tương lai 3 bước (3-Periods Ahead)
    FORECAST_HORIZON = 3
    forecast_dates = pd.date_range(start=target_series.index[-1] + pd.DateOffset(days=1), periods=FORECAST_HORIZON, freq="D")
    sarima_forecast = sarima_model.forecast(steps=FORECAST_HORIZON)
    sarima_forecast.index = forecast_dates
    
    xgb_history = list(target_series.values)
    xgb_forecasts = []
    for step in range(FORECAST_HORIZON):
        row = pd.DataFrame([[xgb_history[-1], xgb_history[-2], xgb_history[-3], np.mean(xgb_history[-3:]), np.std(xgb_history[-3:]), (target_series.index[-1].month + step)%12 + 1, ((target_series.index[-1].month + step)%12)//3 + 1, len(xgb_history)+step]], columns=X_train.columns)
        pred = float(xgb_model.predict(row)[0])
        xgb_forecasts.append(pred)
        xgb_history.append(pred)
    xgb_forecast_series = pd.Series(xgb_forecasts, index=forecast_dates)
    
    forecast_df = pd.DataFrame({
        "Giai đoạn": forecast_dates.strftime("%Y-%m-%d"),
        "Dự báo SARIMA (°C)": sarima_forecast.round(1).values,
        "Dự báo XGBoost (°C)": xgb_forecast_series.round(1).values
    })
    
    return df, daily_total, decomp, y_test, results, forecast_df, xgb_model, X_train

# KÍCH HOẠT PIPELINE DỮ LIỆU
pipeline_data = run_data_science_pipeline()

if pipeline_data is None:
    st.error(f"🚨 Không tìm thấy file dữ liệu: '{DATA_FILENAME}' tại nhánh GitHub của bạn!")
else:
    df, daily_total, decomp, y_test, results, forecast_df, xgb_model, X_train = pipeline_data

    # ─── THANH ĐIỀU HƯỚNG SIDEBAR ───
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4217/4217169.png", width=65)
    st.sidebar.markdown(f"<h3 style='color:{BRAND_CLR}; margin-top:0;'>Analytics Menu</h3>", unsafe_allow_html=True)
    st.sidebar.write("---")
    
    st.sidebar.subheader("🎛️ Bộ lọc hiển thị")
    device_list = ["Tất cả thiết bị"] + list(df["Mã_Thiết_Bị"].unique())
    selected_dev = st.sidebar.selectbox("Lựa chọn Trạm máy IoT", device_list)
    
    df_filtered = df.copy()
    if selected_dev != "Tất cả thiết bị":
        df_filtered = df[df["Mã_Thiết_Bị"] == selected_dev]

    # ─── GIAO DIỆN CHÍNH WEB DASHBOARD (SECTION 10) ───
    st.markdown(
        f"""
        <div style="background:{BG_GRADIENT}; padding:22px; border-radius:10px; margin-bottom:25px; color:white;">
            <h1 style="margin:0; font-size:2.1rem; font-weight:700;">📈 ABC MANUFACTURING — DEMAND FORECASTING SOLUTION</h1>
            <p style="margin:4px 0 0 0; opacity:0.85; font-size:1rem;">Hệ thống phân tích Khoa học dữ liệu & Mô hình hóa chuỗi thời gian dự báo nâng cao</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # HIỂN THỊ CÁC CHỈ SỐ DOANH NGHIỆP THỰC TẾ (KPI METRICS)
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.metric("Tổng bản ghi IoT", f"{len(df_filtered):,}")
    with kpi_col2:
        st.metric("Nhiệt độ Nhà máy Avg", f"{df_filtered['Nhiệt_Độ_C'].mean():.2f} °C")
    with kpi_col3:
        st.metric("Điện năng Tiêu thụ Avg", f"{df_filtered['Điện_Năng_kWh'].mean():.1f} kWh")
    with kpi_col4:
        st.metric("Sự cố ngoại lai (Outliers)", f"{df_filtered['is_outlier'].sum()} dòng")

    st.markdown("<br>", unsafe_allow_html=True)

    # PHÂN TÁCH CÁC PHẦN THEO TAB GIỐNG CẤU TRÚC ĐỀ BÀI
    tab_eda, tab_model, tab_report = st.tabs([
        "📊 Phân Tích Khám Phá Cảm Biến (Section 4)", 
        "🤖 Mô Hình Hóa Học Máy & Dự Báo (Section 6-8)", 
        "📋 Báo Cáo Điều Hành Trưởng (Section 10)"
    ])
    
    # ─── TAB 1: SECTION 4 (EDA VISUALISATION) ───
    with tab_eda:
        st.markdown("### Khám Phá Xu Hướng Biến Động Thời Gian & Mùa Vụ")
        col_e1, col_e2 = st.columns(2)
        
        with col_e1:
            fig_e1, ax_e1 = plt.subplots(figsize=(10, 4.5))
            ax_e1.plot(daily_total["date"], daily_total["avg_temp"], color=PALETTE[0], linewidth=2)
            ax_e1.fill_between(daily_total["date"], daily_total["avg_temp"], alpha=0.15, color=PALETTE[0])
            ax_e1.set_title("Biến Động Nhiệt Độ Trung Bình Toàn Nhà Máy")
            st.pyplot(fig_e1)
            
        with col_e2:
            fig_e2, ax_e2 = plt.subplots(figsize=(10, 4.5))
            decomp.seasonal.plot(ax=ax_e2, color=PALETTE[2], linewidth=1.5)
            ax_e2.set_title("Thành Phần Mùa Vụ Tuần Hoàn (Weekly Seasonal Component)")
            st.pyplot(fig_e2)
            
        # Ma trận bản đồ nhiệt mối tương quan thiết bị
        st.markdown("#### Ma trận Phân phối Mật độ Nhiệt năng theo Thiết bị")
        fig_heat, ax_heat = plt.subplots(figsize=(14, 4))
        pivot_heat = df.groupby(["Mã_Thiết_Bị", "Trạng_Thái_Vận_Hành"])["Nhiệt_Độ_C"].mean().unstack().fillna(0)
        sns.heatmap(pivot_heat, ax=ax_heat, cmap="Blues", annot=True, fmt=".1f", linewidths=0.5)
        st.pyplot(fig_heat)

    # ─── TAB 2: SECTION 6, 7 & 8 (MODELING & FORECASTING) ───
    with tab_model:
        st.markdown("### Đối Chiếu Thực Tế Và Các Mô Hình Kiểm Định (Test Set)")
        
        # Đồ thị so sánh các mô hình chuỗi thời gian nâng cao
        fig_m1, ax_m1 = plt.subplots(figsize=(15, 5))
        ax_m1.plot(y_test.index, y_test.values, "o-", color="black", label="Dữ liệu Thực Tế", zorder=5)
        colors_m = [PALETTE[0], PALETTE[1], PALETTE[3]]
        for idx, (name, res) in enumerate(results.items()):
            ax_m1.plot(res["preds"].index, res["preds"].values, "--", color=colors_m[idx], label=f"{name} (R²={res['R2']:.2f})")
        ax_m1.set_title("So Sánh Hiệu Năng Mô Hình trên Tập Kiểm Thử (Validation Window)")
        ax_m1.legend()
        st.pyplot(fig_m1)
        
        col_m1, col_m2 = st.columns([4, 6])
        with col_m1:
            st.markdown("#### Bảng Đo Lường Sai Số Thuật Toán")
            eval_rows = []
            for name, res in results.items():
                eval_rows.append({"Mô hình": name, "Sai số (MAE)": round(res["MAE"], 2), "Độ tương quan (R²)": round(res["R2"], 2)})
            st.dataframe(pd.DataFrame(eval_rows), use_container_width=True, hide_index=True)
            
            st.markdown("🏆 **Kết luận:** Mô hình có chỉ số R² cao nhất là mô hình tối ưu được lựa chọn cho hệ thống.")
            
        with col_m2:
            st.markdown("#### Kết Quả Dự Báo Cho 3 Giai Đoạn Tiếp Theo (Next 3-Periods)")
            st.dataframe(forecast_df.style.highlight_max(axis=1, color="#E0F2FE"), use_container_width=True, hide_index=True)

    # ─── TAB 3: SECTION 10 (EXECUTIVE SUMMARY REPORT & EXPORT) ───
    with tab_report:
        st.markdown("### Khung Báo Cáo Tổng Hợp Điều Hành (Executive Summary)")
        
        col_r1, col_r2 = st.columns([6, 4])
        with col_r1:
            st.markdown(
                f"""
                <div style="background-color: #F8FAFC; padding: 20px; border-radius: 8px; border-left: 5px solid #1E3A5F; line-height: 1.6;">
                    <h4 style="color: #1E3A5F; margin-top: 0; font-weight:700;">📋 KHUYẾN NGHỊ VẬN HÀNH DÀNH CHO GIÁM ĐỐC NHÀ MÁY (OPERATION DIRECTOR)</h4>
                    <ol>
                        <li><b>Triển khai Thuật toán:</b> Ứng dụng mô hình XGBoostRegressor làm lõi dự báo biến động phụ tải nhiệt năng hàng tuần của trạm SMT.</li>
                        <li><b>Quản lý Dự phòng rủi ro:</b> Tăng cường hành lang an toàn tản nhiệt cho các thiết bị thuộc nhóm cảnh báo lỗi tần suất cao.</li>
                        <li><b>Tối ưu Hóa Chi Phí:</b> Dựa vào biểu đồ phân tách thành phần mùa vụ ở Tab EDA để lên lịch bảo dưỡng máy vào chu kỳ thấp điểm, tiết kiệm năng lượng.</li>
                        <li><b>Tái huấn luyện định kỳ:</b> Cập nhật liên tục Nhật ký vận hành từ file Excel vào Dashboard này mỗi quý một lần để duy trì độ tin cậy của AI.</li>
                    </ol>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col_r2:
            st.markdown("#### Trọng Số Quan Trọng Tính Năng (XGBoost Feature Importance)")
            fig_fi, ax_fi = plt.subplots(figsize=(10, 5))
            feat_imp = pd.Series(xgb_model.feature_importances_, index=X_train.columns).sort_values(ascending=True)
            feat_imp.plot(kind="barh", ax=ax_fi, color="#2563EB")
            st.pyplot(fig_fi)
            
        st.write("---")
        st.markdown("#### 📂 Trích Xuất Dữ Liệu")
        # Nút xuất file dữ liệu sạch
        csv_data = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Tải tệp dữ liệu đã làm sạch (.CSV)",
            data=csv_data,
            file_name='abc_manufacturing_cleaned.csv',
            mime='text/csv',
        )
