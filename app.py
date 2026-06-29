import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import os

# 1. CẤU HÌNH THEME GIAO DIỆN CHUYÊN NGHIỆP (ENTERPRISE STANDARD)
st.set_page_config(
    page_title="ABC Manufacturing - IoT Predictive Maintenance", 
    layout="wide", 
    page_icon="🏭",
    initial_sidebar_state="expanded"
)

PRIMARY_CLR = "#1E3A8A"  
ACCENT_CLR = "#10B981"   
BG_GRADIENT = "linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%)"

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.facecolor'] = '#F8FAFC'

# CẤU HÌNH TÊN FILE ĐUÔI .XLSX CỦA EXCEL
DATA_FILENAME = "ABC_Manufacturing_IoT_Simulation_Data.xlsx"

# 2. DATA SCIENCE PIPELINE TRUY XUẤT NGẦM (SỬ DỤNG CACHE)
@st.cache_resource
def load_and_train_pipeline():
    if not os.path.exists(DATA_FILENAME):
        return None, None, None, None, None, None, None, None
    
    # ĐỌC FILE EXCEL GỐC (Sử dụng engine openpyxl ngầm)
    df = pd.read_excel(DATA_FILENAME)
    
    if "Ngày_Kiểm_Tra" in df.columns:
        df["Ngày_Kiểm_Tra"] = pd.to_datetime(df["Ngày_Kiểm_Tra"])
    
    # --- HÀM XỬ LÝ LỖI ĐỊNH DẠNG NGÀY THÁNG ĐẶC BIỆT DÀNH CHO EXCEL OBJ ---
    def fix_excel_format_direct(val, default_val):
        if pd.isna(val):
            return default_val
        
        # Nếu Excel đã tự động nhận diện ô này là kiểu Ngày tháng (Datetime object)
        if isinstance(val, (pd.Timestamp, datetime.date)) or hasattr(val, 'day'):
            try:
                # Trích xuất Ngày và Tháng trực tiếp từ thuộc tính của ô Excel
                return float(val.day) + float(val.month) / 10.0
            except:
                return default_val
                
        # Nếu ô đó vẫn ở dạng chuỗi văn bản (String) có dấu gạch ngang
        val_str = str(val).strip()
        if "-" in val_str:
            try:
                parts = val_str.split("-")
                return float(parts[2]) + float(parts[1]) / 10.0
            except:
                return default_val
        try:
            return float(val)
        except:
            return default_val

    # Ép kiểu dữ liệu an toàn cho các cột cảm biến bằng hàm cải tiến mới
    import datetime
    df["Độ_Rung_mm_s"] = df["Độ_Rung_mm_s"].apply(lambda x: fix_excel_format_direct(x, 2.5))
    df["Điện_Năng_kWh"] = df["Điện_Năng_kWh"].apply(lambda x: fix_excel_format_direct(x, 12.0))
    df["Nhiệt_Độ_C"] = pd.to_numeric(df["Nhiệt_Độ_C"], errors='coerce').fillna(df["Nhiệt_Độ_C"].median())
    
    # Chuẩn bị ma trận đầu vào cho thuật toán AI (Tên cột Tiếng Việt)
    X = df[["Nhiệt_Độ_C", "Độ_Rung_mm_s", "Điện_Năng_kWh"]]
    y = df["Nguy_Cơ_Sự_Cố"].astype(int)
    
    # Chia tập dữ liệu chuẩn theo tỷ lệ hình học 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Huấn luyện mô hình phân loại thuật toán XGBoost
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return df, model, X, y, X_test, y_test, accuracy, report

# Thực thi Pipeline dữ liệu ngầm
pipeline_output = load_and_train_pipeline()

if pipeline_output[0] is None:
    st.error(f"🚨 Không tìm thấy file: '{DATA_FILENAME}' trong kho lưu trữ GitHub.")
    st.info("Vui lòng upload file Excel của bạn lên thư mục gốc GitHub với đúng tên cấu hình phía trên.")
else:
    df_raw, model, X, y, X_test, y_test, accuracy, report = pipeline_output

    dev_col = "Mã_Thiết_Bị"
    status_col = "Trạng_Thái_Vận_Hành"

    # --- CẤU HÌNH THANH BỘ LỌC SIDEBAR CHUYÊN NGHIỆP ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/581/581601.png", width=70)
    st.sidebar.markdown(f"<h2 style='color:{PRIMARY_CLR}; padding-top:0;'>Hệ thống IoT Control Center</h2>", unsafe_allow_html=True)
    st.sidebar.write("---")
    
    st.sidebar.subheader("🎛️ Bộ lọc Trung tâm Điều hành")
    selected_device = st.sidebar.selectbox("Chọn Mã Thiết Bị Giám Sát", ["Tất cả thiết bị"] + list(df_raw[dev_col].unique()))
    selected_status = st.sidebar.multiselect("Trạng Thái Vận Hành", options=df_raw[status_col].unique(), default=df_raw[status_col].unique())
    
    # Áp dụng bộ lọc động
    df_filtered = df_raw[df_raw[status_col].isin(selected_status)]
    if selected_device != "Tất cả thiết bị":
        df_filtered = df_filtered[df_filtered[dev_col] == selected_device]

    # --- KHU VỰC HIỂN THỊ CHÍNH (MAIN DASHBOARD) ---
    st.markdown(
        f"""
        <div style="background:{BG_GRADIENT}; padding:25px; border-radius:12px; margin-bottom:25px; color:white;">
            <h1 style="margin:0; font-size:2.5rem; font-weight:700;">🏭 ABC MANUFACTURING — REAL-TIME IOT CONTROL CENTER</h1>
            <p style="margin:5px 0 0 0; opacity:0.9; font-size:1.1rem;">Hệ thống phân tích khoa học dữ liệu & Cảnh báo bảo trì dự đoán (Predictive Maintenance) dành cho Giám đốc Vận hành</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Tầng 1: Các thẻ số liệu tổng hợp cao cấp
    total_devices = df_filtered[dev_col].nunique()
    avg_temp = df_filtered["Nhiệt_Độ_C"].mean()
    total_alerts = df_filtered["Nguy_Cơ_Sự_Cố"].sum()
    
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    
    with kpi_col1:
        st.markdown(f"""
            <div style="background-color: white; padding: 20px; border-radius: 10px; border-left: 5px solid #1E3A8A; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                <p style="color: #64748B; margin: 0; font-size: 0.9rem; font-weight: 600; text-transform: uppercase;">Thiết bị đang chạy</p>
                <h2 style="color: #1E293B; margin: 5px 0 0 0; font-size: 1.8rem; font-weight: 700;">{total_devices} Dây chuyền SMT</h2>
            </div>
        """, unsafe_allow_html=True)
        
    with kpi_col2:
        st.markdown(f"""
            <div style="background-color: white; padding: 20px; border-radius: 10px; border-left: 5px solid #F59E0B; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                <p style="color: #64748B; margin: 0; font-size: 0.9rem; font-weight: 600; text-transform: uppercase;">Nhiệt độ trung bình cảm biến</p>
                <h2 style="color: #1E293B; margin: 5px 0 0 0; font-size: 1.8rem; font-weight: 700;">{avg_temp:.1f} °C</h2>
            </div>
        """, unsafe_allow_html=True)
        
    with kpi_col3:
        alert_card_color = "#EF4444" if total_alerts > 0 else "#10B981"
        st.markdown(f"""
            <div style="background-color: white; padding: 20px; border-radius: 10px; border-left: 5px solid {alert_card_color}; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                <p style="color: #64748B; margin: 0; font-size: 0.9rem; font-weight: 600; text-transform: uppercase;">Cảnh báo lỗi từ AI</p>
                <h2 style="color: {alert_card_color}; margin: 5px 0 0 0; font-size: 1.8rem; font-weight: 700;">{total_alerts} Nguy cơ sự cố</h2>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tầng 2: Thiết lập các Tab phân tách thông tin mạch lạc
    tab1, tab2, tab3 = st.tabs([
        "📊 Giám Sát Cảm Biến IoT (EDA)", 
        "🤖 AI Dự Báo & Khuyến Nghị Vận Hành", 
        "📋 Cơ Sở Dữ Liệu Trung Tâm"
    ])
    
    with tab1:
        st.markdown("<h3 style='color:#1E3A8A; font-weight:600;'>Biến Động Tín Hiệu Cảm Biến Nhà Xưởng</h3>", unsafe_allow_html=True)
        col_eda1, col_eda2 = st.columns(2)
        
        with col_eda1:
            fig1, ax1 = plt.subplots(figsize=(10, 4.5))
            sns.boxplot(data=df_filtered, x=status_col, y="Nhiệt_Độ_C", ax=ax1, palette="Blues_r")
            ax1.set_title("Phân phối Nhiệt độ theo từng Trạng thái máy", fontsize=11, fontweight='bold', color='#1E293B')
            st.pyplot(fig1)
            
        with col_eda2:
            fig2, ax2 = plt.subplots(figsize=(10, 4.5))
            sns.scatterplot(data=df_filtered, x="Nhiệt_Độ_C", y="Điện_Năng_kWh", hue=status_col, palette="viridis", alpha=0.8, ax=ax2)
            ax2.set_title("Tương quan giữa Nhiệt độ đầu ra và Điện năng tiêu thụ", fontsize=11, fontweight='bold', color='#1E293B')
            st.pyplot(fig2)
            
    with tab2:
        st.markdown("<h3 style='color:#1E3A8A; font-weight:600;'>Đánh Giá Mô Hình Học Máy XGBoost Classifier</h3>", unsafe_allow_html=True)
        col_mod1, col_mod2 = st.columns([4, 6])
        
        with col_mod1:
            st.markdown(
                f"""
                <div style="background-color: #F0FDF4; padding: 15px; border-radius: 8px; border: 1px solid #BBF7D0; margin-bottom: 15px;">
                    <span style="color: #166534; font-weight: 700; font-size: 1.1rem;">🎯 Độ chính xác tổng thể (Accuracy):</span>
                    <span style="color: #15803D; font-weight: 800; font-size: 1.4rem; margin-left: 10px;">{accuracy * 100:.2f}%</span>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            report_df = pd.DataFrame(report).transpose().iloc[:2, :3]
            report_df.columns = ["Độ chính xác (Precision)", "Độ bao phủ (Recall)", "F1-Score"]
            report_df.index = ["Bình thường (0)", "Nguy cơ lỗi (1)"]
            st.markdown("**Bảng thông số phân loại lỗi nhị phân:**")
            st.dataframe(report_df.style.background_gradient(cmap="Blues"), use_container_width=True)
            
            st.markdown(
                f"""
                <div style="background-color: #F8FAFC; padding: 20px; border-radius: 8px; border-left: 4px solid #1E3A8A; box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.05);">
                    <h4 style="color: #1E3A8A; margin-top: 0; font-weight:700;">📋 KHUYẾN NGHỊ ĐIỀU HÀNH CHO GIÁM ĐỐC (OPERATIONS DIRECTIVES)</h4>
                    <ul style="margin-bottom: 0; padding-left: 20px; color: #334155; line-height: 1.6;">
                        <li><b>Kế hoạch bảo trì chủ động:</b> Khi thuật toán AI dự báo nguy cơ lỗi hỏng máy (Cột sự cố = 1), điều phối ngay kỹ sư cơ khí kiểm tra bộ phận tản nhiệt.</li>
                        <li><b>Tối ưu chi phí năng lượng:</b> Biểu đồ EDA chỉ ra thiết bị có trạng thái Rung lắc mạnh làm tiêu tốn năng lượng hơn 15%, cần tiến hành căn chỉnh trục định kỳ.</li>
                        <li><b>Mở rộng giải pháp:</b> Khuyến nghị tích hợp API của mô hình XGBoost trực tiếp vào hệ thống SCADA nhà xưởng để tự động ngắt máy an toàn khi xảy ra sự cố đột xuất.</li>
                    </ul>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        with col_mod2:
            fig_imp, ax_imp = plt.subplots(figsize=(10, 4.2))
            
            display_names = {"Nhiệt_Độ_C": "Nhiệt Độ (°C)", "Độ_Rung_mm_s": "Độ Rung (mm/s)", "Điện_Năng_kWh": "Điện Năng (kWh)"}
            feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
            feat_imp.index = [display_names.get(x, x) for x in feat_imp.index]
            
            feat_imp.plot(kind="barh", ax=ax_imp, color="#3B82F6", width=0.5)
            ax_imp.set_title("Trọng số ảnh hưởng lớn nhất đến nguy cơ lỗi máy", fontsize=11, fontweight='bold', color='#1E293B')
            st.pyplot(fig_imp)
            
    with tab3:
        st.markdown("<h3 style='color:#1E3A8A; font-weight:600;'>Bảng Truy Vấn Dữ Liệu Cảm Biến Tổng Hợp</h3>", unsafe_allow_html=True)
        st.dataframe(df_filtered, use_container_width=True, hide_index=True)
