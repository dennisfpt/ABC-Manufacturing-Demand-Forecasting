"""
============================================================
 ABC MANUFACTURING — DEMAND FORECASTING WEB SOLUTION
 Data Science Case Study | Operation Director Support
 Author  : Junior Analyst, ABC Manufacturing
 Version : 2.5 (Excel .xlsx Native Integration)
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
import xgboost as xgb
import os
import datetime

# 1. CẤU HÌNH GIAO DIỆN WEB STREAMLIT
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

# ĐỊNH NGHĨA TÊN FILE EXCEL TRÊN GITHUB CỦA BẠN
DATA_FILENAME = "ABC_Manufacturing_IoT_Simulation_Data.xlsx"

# =============================================================================
# DATA PIPELINE (ĐỌC FILE EXCEL & XỬ LÝ LỖI ĐỊNH DẠNG Ô TỰ ĐỘNG)
# =============================================================================
@st.cache_resource
def run_data_science_pipeline():
    if not os.path.exists(DATA_FILENAME):
        return None
        
    # Đọc trực tiếp file Excel cấu trúc .xlsx bằng engine openpyxl
    try:
        df_raw = pd.read_excel(DATA_FILENAME, engine='openpyxl')
    except Exception:
        # Dự phòng nếu môi trường GitHub của bạn chưa nhận diện đúng định dạng
        df_raw = pd.read_excel(DATA_FILENAME)
        
    df = df_raw.copy()
    df["Ngày_Kiểm_Tra"] = pd.to_datetime(df["Ngày_Kiểm_Tra"])
    
    # Hàm xử lý chuẩn hóa các ô bị lỗi biến thành định dạng Ngày (Date Glitch) trong Excel
    def clean_excel_date_glitch(val, default_numeric_val):
        if pd.isna(val): 
            return default_numeric_val
        # Nếu ô bị biến thành đối tượng ngày tháng (Timestamp)
        if isinstance(val, (pd.Timestamp, datetime.date)) or hasattr(val, 'day'):
            try: 
                return float(val.day) + float(val.month) / 10.0
            except: 
                return default_numeric_val
        # Nếu ô ở dạng chuỗi văn bản chứa dấu gạch ngang (VD: "2026-01-02")
        val_str = str(val).strip()
        if "-" in val_str:
            try:
                parts = val_str.split("-")
                return float(parts[2]) + float(parts[1]) / 10.0
            except: 
                return default_numeric_val
        try: 
            return float(val)
        except: 
            return default_numeric_val

    # Áp dụng hàm làm sạch cho hai cột bị lỗi định dạng trên file Excel thực tế
    df["Độ_Rung_mm_s"] = df["Độ_Rung_mm_s"].apply(lambda x: clean_excel_date_glitch(x, 2.5))
    df["Điện_Năng_kWh"] = df["Điện_Năng_kWh"].apply(lambda x: clean_excel_date_glitch(x, 12.0))
    df["Nhiệt_Độ_C"] = pd.to_numeric(df["Nhiệt_Độ_C"], errors='coerce').fillna(df["Nhiệt_Độ_C"].median())
    
    df["year"]    = df["Ngày_Kiểm_Tra"].dt.year
    df["month"]   = df["Ngày_Kiểm_Tra"].dt.month
    df["quarter"] = df["Ngày_Kiểm_Tra"].dt.quarter

    # Tìm kiếm các điểm dị biệt (Outliers) bằng phương pháp IQR phục vụ báo cáo giám đốc
    q1, q3 = df["Nhiệt_Độ_C"].quantile(0.25), df["Nhiệt_Độ_C"].quantile(0.75)
    iqr = q3 - q1
    df["is_outlier"] = ~df["Nhiệt_Độ_C"].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    
    # Phân tách chuỗi thời gian (Seasonal Decompose) cho toàn nhà máy
    daily_total = df.groupby("Ngày_Kiểm_Tra")["Nhiệt_Độ_C"].mean().reset_index()
    daily_total.columns = ["date", "avg_temp"]
    decomp = seasonal_decompose(daily_total.set_index("date")["avg_temp"], model="additive", period=2, extrapolate_trend="freq")
    
    # Feature Engineering tập trung vào thiết bị cốt lõi SMT-001
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
    
    # Chia dữ liệu huấn luyện độc lập Train/Test (Tỷ lệ 80/20)
    SPLIT = int(len(feat_df) * 0.80) if len(feat_df) > 5 else len(feat_df) - 1
    X_train, X_test = feat_df.iloc[:SPLIT].drop("y", axis=1), feat_df.iloc[SPLIT:].drop("y", axis=1)
    y_train, y_test = feat_df.iloc[:SPLIT]["y"], feat_df.iloc[SPLIT:]["y"]
    
    # TIẾN HÀNH HUẤN LUYỆN 3 MÔ HÌNH KIỂM ĐỊNH
    results = {}
    
    # Mô hình 1: Baseline Moving Average
    baseline_pred = target_series.shift(1).rolling(2, min_periods=1).mean().reindex(y_test.index)
    results["Baseline (MA-3)"] = {"preds": baseline_pred, "MAE": mean_absolute_error(y_test, baseline_pred), "R2": r2_score(y_test, baseline_pred)}
    
    # Mô hình 2: SARIMA Model
    train_ts = target_series.iloc[: SPLIT + len(feat_df) - len(target_series) + 3]
    sarima_model = SARIMAX(train_ts, order=(1,1,1), seasonal_order=(0,0,0,0), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarima_pred = sarima_model.predict(start=y_test.index[0], end=y_test.index[-1])
    results["SARIMA Model"] = {"preds": sarima_pred, "MAE": mean_absolute_error(y_test, sarima_pred), "R2": r2_score(y_test, sarima_pred)}
    
    # Mô hình 3: Machine Learning XGBoost Regressor
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train)
    xgb_pred = pd.Series(xgb_model.predict(X_test), index=y_test.index)
    results["XGBoost Regressor"] = {"preds": xgb_pred, "MAE": mean_absolute_error(y_test, xgb_pred), "R2": r2_score(y_test, xgb_pred)}
    
    # THỰC HIỆN DỰ BÁO TƯƠNG LAI 3 BƯỚC (3-Periods Ahead)
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
        "Ngày tiếp theo": forecast_dates.strftime("%Y-%m-%d"),
        "Mô hình SARIMA (°C)": sarima_forecast.round(1).values,
        "Mô hình XGBoost (°C)": xgb_forecast_series.round(1).values
    })
    
    return df, daily_total, decomp, y_test, results, forecast_df, xgb_model, X_train

# KÍCH HOẠT ĐƯỜNG ỐNG DỮ LIỆU CHẠY NGẦM
pipeline_data = run_data_science_pipeline()

if pipeline_data is None:
    st.error(f"🚨 Hệ thống không tìm thấy file Excel tên là '{DATA_FILENAME}' trong kho lưu trữ GitHub của bạn!")
else:
    df, daily_total, decomp, y_test, results, forecast_df, xgb_model, X_train = pipeline_data

    # ─── THANH ĐIỀU HƯỚNG BÊN (SIDEBAR) ───
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4217/4217169.png", width=65)
    st.sidebar.markdown(f"<h3 style='color:{BRAND_CLR}; margin-top:0;'>Bảng Điều Khiển</h3>", unsafe_allow_html=True)
    st.sidebar.write("---")
    
    st.sidebar.subheader("🎛️ Bộ lọc trạm máy")
    device_list = ["Tất cả thiết bị"] + list(df["Mã_Thiết_Bị"].unique())
    selected_dev = st.sidebar.selectbox("Lựa chọn Thiết bị đầu cuối IoT", device_list)
    
    df_filtered = df.copy()
    if selected_dev != "Tất cả thiết bị":
        df_filtered = df[df["Mã_Thiết_Bị"] == selected_dev]

    # ─── GIAO DIỆN CHÍNH (MAIN SCREEN) ───
    st.markdown(
        f"""
        <div style="background:{BG_GRADIENT}; padding:22px; border-radius:10px; margin-bottom:25px; color:white;">
            <h1 style="margin:0; font-size:2.1rem; font-weight:700;">📈 ABC MANUFACTURING — DEMAND FORECASTING WEB DASHBOARD</h1>
            <p style="margin:4px 0 0 0; opacity:0.85; font-size:1rem;">Hệ thống phân tích chuỗi thời gian & Mô hình hóa dự báo phụ tải cảm biến từ dữ liệu Excel thực tế</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # THẺ CHỈ SỐ DOANH NGHIỆP CỐT LÕI (KPI METRICS)
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.metric("Tổng số dòng dữ liệu Excel", f"{len(df_filtered):,}")
    with kpi_col2:
        st.metric("Nhiệt độ Nhà máy Avg", f"{df_filtered['Nhiệt_Độ_C'].mean():.1f} °C")
    with kpi_col3:
        st.metric("Điện năng Tiêu thụ Avg", f"{df_filtered['Điện_Năng_kWh'].mean():.1f} kWh")
    with kpi_col4:
        st.metric("Điểm Dị biệt Hệ thống (Outliers)", f"{df_filtered['is_outlier'].sum()} dòng")

    st.markdown("<br>", unsafe_allow_html=True)

    # PHÂN TÁCH GIAO DIỆN THÀNH CÁC TAB CHUYÊN CHÚNG
    tab_eda, tab_model, tab_report = st.tabs([
        "📊 Phân Tích Thống Kê Khám Phá (Section 4)", 
        "🤖 Kiểm Định Mô Hình Học Máy (Section 6-8)", 
        "📋 Báo Cáo Hội Đồng Quản Trị (Section 10)"
    ])
    
    # ─── TAB 1: BIỂU ĐỒ PHÂN TÍCH CHUỖI THỜI GIAN ───
    with tab_eda:
        st.markdown("### Phân tích Xu hướng Chuỗi Thời gian & Thành phần Biến động")
        col_e1, col_e2 = st.columns(2)
        
        with col_e1:
            fig_e1, ax_e1 = plt.subplots(figsize=(10, 4.5))
            ax_e1.plot(daily_total["date"], daily_total["avg_temp"], color=PALETTE[0], linewidth=2, marker='o')
            ax_e1.fill_between(daily_total["date"], daily_total["avg_temp"], alpha=0.15, color=PALETTE[0])
            ax_e1.set_title("Đồ thị Giám sát Nhiệt độ Diễn tiến Toàn Nhà máy")
            ax_e1.set_ylabel("Nhiệt độ (°C)")
            st.pyplot(fig_e1)
            
        with col_e2:
            fig_e2, ax_e2 = plt.subplots(figsize=(10, 4.5))
            decomp.trend.plot(ax=ax_e2, color=PALETTE[1], linewidth=2, marker='s')
            ax_e2.set_title("Thành phần Xu hướng cốt lõi (Decomposed Trend Element)")
            st.pyplot(fig_e2)
            
        st.markdown("#### Ma trận Tương quan Giữa Thiết bị & Trạng thái Vận hành Thực tế")
        fig_heat, ax_heat = plt.subplots(figsize=(14, 4))
        pivot_heat = df.groupby(["Mã_Thiết_Bị", "Trạng_Thái_Vận_Hành"])["Nhiệt_Độ_C"].mean().unstack().fillna(0)
        sns.heatmap(pivot_heat, ax=ax_heat, cmap="Blues", annot=True, fmt=".1f", linewidths=0.5, cbar_kws={'label': 'Nhiệt độ trung bình'})
        st.pyplot(fig_heat)

    # ─── TAB 2: ĐỐI CHIẾU VÀ DỰ BÁO TƯƠNG LAI ───
    with tab_model:
        st.markdown("### Kiểm thử Mô hình Dự báo trên Vùng Dữ liệu Chứng thực (Test Range)")
        
        fig_m1, ax_m1 = plt.subplots(figsize=(15, 4.5))
        ax_m1.plot(y_test.index, y_test.values, "o-", color="black", linewidth=2.5, label="Dữ liệu Thực tế (Actual)")
        colors_m = [PALETTE[0], PALETTE[1], PALETTE[3]]
        for idx, (name, res) in enumerate(results.items()):
            ax_m1.plot(res["preds"].index, res["preds"].values, "x--", color=colors_m[idx], label=f"{name} (R²: {res['R2']:.2f})")
        ax_m1.set_title("So sánh Kết quả Dự báo Đối chiếu Giữa Các Thuật toán")
        ax_m1.set_ylabel("Nhiệt độ (°C)")
        ax_m1.legend()
        st.pyplot(fig_m1)
        
        col_m1, col_m2 = st.columns([4, 6])
        with col_m1:
            st.markdown("#### 📐 Bảng Chỉ số Đo lường Sai số")
            eval_rows = []
            for name, res in results.items():
                eval_rows.append({
                    "Thuật toán mô hình": name, 
                    "Sai số MAE": round(res["MAE"], 2), 
                    "Hệ số xác định R²": round(res["R2"], 2)
                })
            st.dataframe(pd.DataFrame(eval_rows), use_container_width=True, hide_index=True)
            st.caption("ℹ️ Hệ số R² càng tiệm cận 1.00 thể hiện thuật toán khớp mẫu chuỗi thời gian càng hoàn hảo.")
            
        with col_m2:
            st.markdown("#### 🔮 Kết quả Dự báo Xu hướng 3 Ngày Kế tiếp")
            st.dataframe(forecast_df.style.highlight_max(axis=0, color="#FEF3C7"), use_container_width=True, hide_index=True)

    # ─── TAB 3: KHUNG BÁO CÁO CHIẾN LƯỢC ĐIỀU HÀNH ───
    with tab_report:
        st.markdown("### Khung Khuyến Nghi Sắp Xếp Lịch Trình Vận Hành")
        
        col_r1, col_r2 = st.columns([6, 4])
        with col_r1:
            st.markdown(
                f"""
                <div style="background-color: #F8FAFC; padding: 22px; border-radius: 8px; border-left: 5px solid #1E3A5F; line-height: 1.7;">
                    <h4 style="color: #1E3A5F; margin-top: 0; font-weight:700;">📋 ĐỀ XUẤT CHO GIÁM ĐỐC ĐIỀU HÀNH (OPERATION DIRECTOR)</h4>
                    <ul>
                        <li><b>Lựa chọn Mô hình cốt lõi:</b> Áp dụng thuật toán <b>XGBoost Regressor</b> tích hợp vào hệ thống SCADA để tự động dự báo phụ tải biên nhà máy.</li>
                        <li><b>Kế hoạch Vật tư Phòng ngừa:</b> Tiến hành kiểm tra trục cơ và cuộn dây đồng khi phát hiện chỉ số Độ rung vượt ngưỡng nền tự động hóa.</li>
                        <li><b>Tối ưu Năng lượng tiêu thụ:</b> Sử dụng kết quả dự báo thời gian thực từ Tab 2 để điều chỉnh công suất trạm làm mát hệ thống vào giờ cao điểm của phụ tải nhiệt.</li>
                        <li><b>Chu kỳ đồng bộ dữ liệu:</b> Đẩy file Excel cập nhật nhật ký cảm biến mới lên kho lưu trữ GitHub định kỳ mỗi quý để mô hình tự học lại hành vi nhiễu.</li>
                    </ul>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col_r2:
            st.markdown("#### Đánh giá Trọng số Thuộc tính (XGBoost Feature Importance)")
            fig_fi, ax_fi = plt.subplots(figsize=(10, 5))
            feat_imp = pd.Series(xgb_model.feature_importances_, index=X_train.columns).sort_values(ascending=True)
            feat_imp.plot(kind="barh", ax=ax_fi, color="#10B981")
            ax_fi.set_xlabel("Điểm trọng số tầm quan trọng")
            st.pyplot(fig_fi)
            
        st.write("---")
        st.markdown("#### 💾 Xuất Dữ Liệu Sạch (Processed Output Sync)")
        csv_buffer = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Tải Tệp Dữ Liệu Sạch Về Máy (.CSV)",
            data=csv_buffer,
            file_name='abc_manufacturing_cleaned.csv',
            mime='text/csv',
        )
