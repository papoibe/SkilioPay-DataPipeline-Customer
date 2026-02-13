# SkilioPay Churn Prediction Dashboard
# 
# Dashboard chính hiển thị KPIs, biểu đồ phân tích churn, 
# model performance, và tra cứu user.
# 
# Chạy: streamlit run src/dashboard/app.py
import streamlit as st
import sys
import os
from pathlib import Path

# Thêm project root vào path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import Data Loader
from src.dashboard.data_loader import (
    load_processed_data, load_model, load_raw_data
)

# Import Views
from src.dashboard.views.pages import (
    render_overview_page, render_analysis_page, 
    render_model_performance_page, render_user_lookup_page, 
    render_system_status_page
)

# PAGE CONFIG
st.set_page_config(
    page_title="SkilioPay Churn Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LOAD CSS
css_file = Path(__file__).parent / "styles" / "main.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# DATA LOADING
@st.cache_data(ttl=600)
def load_data():
    return load_processed_data()

@st.cache_data(ttl=600)
def load_raw_data_cached():
    return load_raw_data()

@st.cache_resource
def load_model_cached():
    return load_model()

# TOP NAVIGATION (Thay thế Sidebar)
with st.container():
    col_logo, col_nav = st.columns([1, 5])
    with col_logo:
        # Logo placeholder
        st.write("SkilioPay")
    with col_nav:
        # Sử dụng Radio button nằm ngang làm menu
        page = st.radio(
            "Menu",
            ["Tổng Quan", "Phân Tích Chi Tiết", "Tra Cứu User", "Hiệu Quả Mô Hình", "Trạng Thái Hệ Thống"],
            index=0,
            horizontal=True,
            label_visibility="collapsed"
        )
    st.markdown("---")

# SIDEBAR (Chỉ còn bộ lọc - Optional)
# with st.sidebar:
#     st.markdown("### Bộ Lọc")
#     st.info("Dữ liệu đang được phân tích trên toàn bộ tập khách hàng.")

# MAIN CONTENT
# Load Data
df = load_data()
df_raw = load_raw_data_cached()
model_data = load_model_cached()

if df is None:
    st.error("Không tìm thấy dữ liệu processed. Vui lòng kiểm tra lại Pipeline.")
    st.stop()

# Route to Pages
if page == "Tổng Quan":
    render_overview_page(df, df_raw)

elif page == "Phân Tích Chi Tiết":
    render_analysis_page(df, df_raw)

elif page == "Hiệu Quả Mô Hình":
    render_model_performance_page(model_data, df)

elif page == "Tra Cứu User":
    render_user_lookup_page(df, df_raw, model_data)

elif page == "Trạng Thái Hệ Thống":
    render_system_status_page(df, df_raw)
