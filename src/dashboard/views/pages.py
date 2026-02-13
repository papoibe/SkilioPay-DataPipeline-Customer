
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.dashboard.logic.plotting import (
    create_churn_donut_chart, create_age_distribution_chart, 
    create_rfm_boxplot, create_country_churn_chart, 
    create_behavior_scatter, create_correlation_heatmap, 
    create_feature_importance_chart, COLORS
)
from src.dashboard.data_loader import (
    get_churn_summary, get_model_metrics, get_feature_importance, get_pipeline_status
)

# PAGE: TỔNG QUAN
def render_overview_page(df, df_raw):
    st.markdown("# Tổng Quan Churn")
    st.markdown("##### Phân tích tỷ lệ rời bỏ khách hàng SkilioPay")
    
    summary = get_churn_summary(df)
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tổng Khách Hàng", f"{summary['total_users']:,}")
    with col2:
        st.metric("Khách Hàng Trung Thành", f"{summary['retained']:,}")
    with col3:
        st.metric("Khách Hàng Rời Bỏ", f"{summary['churned']:,}")
    with col4:
        st.metric("Tỷ Lệ Rời Bỏ", f"{summary['churn_rate']:.1%}")

    st.markdown("---")

    # Row 2: Charts
    col_left, col_right = st.columns([1, 1.5])
    
    with col_left:
        st.markdown("#### Tỷ Lệ Rời Bỏ")
        fig_donut = create_churn_donut_chart(summary['retained'], summary['churned'], summary['churn_rate'])
        st.plotly_chart(fig_donut, use_container_width=True)
    
    with col_right:
        st.markdown("#### Phân Bố Theo Độ Tuổi")
        # Use Raw Data for age if available
        data_for_age = df_raw if df_raw is not None and 'age' in df_raw.columns else df
        
        if 'age' in data_for_age.columns:
            plot_df = data_for_age.copy()
            churn_col = 'churn_label' if 'churn_label' in plot_df.columns else ('Churn Label' if 'Churn Label' in plot_df.columns else None)
            
            if churn_col:
                plot_df['Trạng Thái'] = plot_df[churn_col].map({0: 'Trung Thành', 1: 'Rời Bỏ'})
                fig_age = create_age_distribution_chart(plot_df)
                st.plotly_chart(fig_age, use_container_width=True)
            else:
                 st.info("Không tìm thấy cột Churn Label trong dữ liệu gốc.")

    # Row 3: RFM
    st.markdown("---")
    st.markdown("#### Phân Tích RFM (Recency, Frequency, Monetary)")
    st.markdown("*RFM giúp đánh giá giá trị khách hàng dựa trên hành vi mua sắm gần đây, tần suất và giá trị đơn hàng.*")
    
    rfm_cols_check = ['rfm_recency', 'rfm_frequency', 'rfm_monetary']
    data_for_rfm = df
    available_rfm = [c for c in rfm_cols_check if c in data_for_rfm.columns]
    
    rfm_names = {
        'rfm_recency': 'Recency (Gần đây)', 
        'rfm_frequency': 'Frequency (Tần suất)', 
        'rfm_monetary': 'Monetary (Chi tiêu)'
    }
    
    if available_rfm:
        cols_rfm = st.columns(len(available_rfm))
        for i, col_name in enumerate(available_rfm):
            with cols_rfm[i]:
                plot_df = data_for_rfm.copy()
                if 'churn_label' in plot_df.columns:
                    plot_df['Trạng Thái'] = plot_df['churn_label'].map({0: 'Trung Thành', 1: 'Rời Bỏ'})
                
                fig_box = create_rfm_boxplot(plot_df, col_name, rfm_names.get(col_name, col_name))
                st.plotly_chart(fig_box, use_container_width=True)

# PAGE: PHÂN TÍCH CHI TIẾT
def render_analysis_page(df, df_raw):
    st.markdown("# Phân Tích Chi Tiết")
    
    tab1, tab2, tab3 = st.tabs(["Theo Quốc Gia", "Hành Vi Mua Hàng", "Mức Độ Tương Tác"])
    
    with tab1:
        if 'country' in df.columns:
            st.markdown("#### Tỷ Lệ Rời Bỏ Theo Quốc Gia")
            country_churn = df.groupby('country').agg(
                total=('churn_label', 'count'),
                churned=('churn_label', 'sum')
            ).reset_index()
            country_churn['churn_rate'] = country_churn['churned'] / country_churn['total']
            country_churn = country_churn.sort_values('churn_rate', ascending=True)
            
            fig_country = create_country_churn_chart(country_churn)
            st.plotly_chart(fig_country, use_container_width=True)
        else:
            st.info("Dữ liệu quốc gia không có sẵn trong processed data.")
    
    with tab2:
        st.markdown("#### Tương Quan Hành Vi Mua Hàng & Rời Bỏ")
        data_for_behavior = df_raw if df_raw is not None else df
        behavior_cols_raw = ['orders_30d', 'orders_90d', 'aov_2024', 'gmv_2024']
        available_behavior = [c for c in behavior_cols_raw if c in data_for_behavior.columns]
        
        if len(available_behavior) >= 2:
            col_x = st.selectbox("Chọn Chỉ Số X (Ngang)", available_behavior, index=0)
            col_y = st.selectbox("Chọn Chỉ Số Y (Dọc)", available_behavior, index=1)
            
            sample_df = data_for_behavior.sample(min(5000, len(data_for_behavior)), random_state=42).copy()
            churn_col = 'churn_label'
            if churn_col not in sample_df.columns and 'Churn Label' in sample_df.columns:
                 churn_col = 'Churn Label'

            if churn_col in sample_df.columns:
                sample_df['Trạng Thái'] = sample_df[churn_col].map({0: 'Trung Thành', 1: 'Rời Bỏ'})
                fig_scatter = create_behavior_scatter(sample_df, col_x, col_y)
                st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab3:
        st.markdown("#### Các Chỉ Số Tương Tác (Engagement)")
        engagement_cols = ['sessions_30d', 'sessions_90d', 'avg_session_duration_90d',
                          'emails_open_rate_90d', 'emails_click_rate_90d']
        available_engagement = [c for c in engagement_cols if c in df.columns]
        
        if available_engagement:
            corr_cols = available_engagement + ['churn_label']
            corr_matrix = df[corr_cols].corr()
            fig_heatmap = create_correlation_heatmap(corr_matrix)
            st.plotly_chart(fig_heatmap, use_container_width=True)

# PAGE: HIỆU QUẢ MÔ HÌNH
def render_model_performance_page(model_data, df):
    st.markdown("# Hiệu Quả Mô Hình")
    st.markdown("##### Đánh giá khả năng dự đoán của hệ thống AI")
    
    if model_data is None:
        st.warning("Chưa có Model. Hãy chạy huấn luyện trước.")
        st.stop()
    
    with st.spinner("Đang tính toán hiệu quả..."):
        metrics = get_model_metrics(model_data, df)
    
    if metrics and 'error' not in metrics:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Độ Chính Xác Tổng Thể", f"{metrics.get('accuracy', 0):.1%}", help="Tỷ lệ dự đoán đúng trên tổng số khách hàng")
        with col2:
            st.metric("Độ Tin Cậy Báo Churn", f"{metrics.get('precision', 0):.1%}", help="Khi AI báo khách sắp bỏ đi, bao nhiêu phần trăm là đúng?")
        with col3:
            st.metric("Khả Năng Phát Hiện Churn", f"{metrics.get('recall', 0):.1%}", help="AI tìm ra được bao nhiêu phần trăm khách hàng thực sự rời bỏ?")
        with col4:
            st.metric("Điểm Chất Lượng (ROC)", f"{metrics.get('roc_auc', 0):.1%}", help="Điểm đánh giá tổng quát sức mạnh phân loại của mô hình (0-100%)")
        
        st.markdown("---")
        
        st.markdown("#### Top Các Yếu Tố Ảnh Hưởng Đến Việc Rời Bỏ")
        importance_df = get_feature_importance(model_data, top_n=15)
        fig_importance = create_feature_importance_chart(importance_df)
        st.plotly_chart(fig_importance, use_container_width=True)

    else:
        if not metrics:
             st.error("Không thể tính toán metrics. Dữ liệu trả về trống (Empty Result).")
        else:
             st.error(f"Lỗi tính toán metrics: {metrics.get('error', 'Unknown Error')}")
        
    with st.expander("Giải thích chi tiết thuật ngữ"):
        st.markdown("""
        - **Độ Chính Xác Tổng Thể**: Cho biết mô hình đúng bao nhiêu trên tổng thể.
        - **Độ Tin Cậy Báo Churn (Precision)**: Quan trọng khi bạn muốn tránh làm phiền khách hàng trung thành.
        - **Khả Năng Phát Hiện Churn (Recall)**: Quan trọng khi bạn không muốn bỏ sót bất kỳ khách hàng nào có nguy cơ rời bỏ.
        - **Điểm Chất Lượng (ROC)**: Điểm số từ 90% trở lên là xuất sắc.
        """)

# PAGE: TRA CỨU USER
def render_user_lookup_page(df, df_raw, model_data):
    st.markdown("# Tra Cứu Khách Hàng")
    st.markdown("##### Nhập User ID để xem hồ sơ và dự đoán rủi ro")
    
    user_id = st.text_input("User ID", placeholder="Nhập ID (Ví dụ: U00102)...").strip()
    
    if user_id:
        # 1. Tìm trong Raw Data
        user_display = None
        if df_raw is not None:
            raw_match = df_raw[df_raw['user_id'] == user_id]
            if not raw_match.empty:
                user_display = raw_match.iloc[0]
        
        # 2. Tìm trong Processed Data
        user_proc = None
        proc_match = df[df['user_id'] == user_id]
        if not proc_match.empty:
            user_proc = proc_match.iloc[0]
        
        if user_display is None and user_proc is None:
            st.warning(f"Không tìm thấy User ID: **{user_id}**")
            if df_raw is not None:
                samples = df_raw['user_id'].head(3).tolist()
                st.info(f"Gợi ý: Thử nhập {', '.join(samples)}")
        else:
            final_display = user_display if user_display is not None else user_proc
            
            st.markdown("#### Hồ Sơ Khách Hàng")
            col1, col2, col3, col4 = st.columns(4)
            
            churn_val = final_display.get('churn_label', final_display.get('Churn Label', 0))
            status_text = "Đã Rời Bỏ" if churn_val == 1 else "Đang Hoạt Động"
            
            with col1:
                st.metric("Trạng Thái", status_text)
            with col2:
                age_val = final_display.get('age', final_display.get('Age', 0))
                st.metric("Tuổi", f"{int(age_val)}" if pd.notna(age_val) else "?")
            with col3:
                reg_val = final_display.get('reg_days', 0)
                st.metric("Thời Gian Gia Nhập", f"{int(reg_val)} ngày")
            with col4:
                ord_val = final_display.get('orders_30d', 0)
                st.metric("Đơn Hàng (30 ngày)", f"{int(ord_val)}")
            
            st.markdown("---")
            st.markdown("#### Dự Đoán Của AI")
            
            if user_proc is not None and model_data:
                feature_cols = model_data['feature_columns']
                scaler = model_data['scaler']
                
                X_pred = proc_match[ [c for c in feature_cols if c in proc_match.columns] ].copy()
                for c in set(feature_cols) - set(X_pred.columns):
                    X_pred[c] = 0
                X_pred = X_pred[feature_cols]
                
                num_cols = X_pred.select_dtypes(include=[np.number]).columns.tolist()
                if scaler:
                    try:
                        X_pred[num_cols] = scaler.transform(X_pred[num_cols])
                    except:
                        pass
                
                try:
                    proba = float(model_data['model'].predict_proba(X_pred)[0][1])
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        risk_level = "Thấp" if proba < 0.3 else ("Trung Bình" if proba < 0.7 else "Cao")
                        st.metric("Xác Suất Rời Bỏ", f"{proba:.1%}")
                        st.write(f"Đánh giá rủi ro: **{risk_level}**")
                    with c2:
                        st.progress(proba)
                except Exception as e:
                    st.error(f"Lỗi dự đoán: {e}")
            else:
                st.warning("Không có dữ liệu Processed cho User này nên không thể Dự đoán.")

            st.markdown("---")
            with st.expander("Xem Dữ Liệu Gốc (Chi Tiết)", expanded=True):
                display_df = final_display.to_frame(name="Giá Trị")
                st.dataframe(display_df, use_container_width=True)

# PAGE: TRẠNG THÁI HỆ THỐNG
def render_system_status_page(df, df_raw):
    st.markdown("# Trạng Thái Hệ Thống")
    
    steps = get_pipeline_status()
    st.table(pd.DataFrame(steps))
    
    st.markdown("### Thông Tin Dữ Liệu")
    col1, col2 = st.columns(2)
    col1.metric("Processed Rows", len(df))
    col1.metric("Processed Columns", len(df.columns))
    
    if df_raw is not None:
        col2.metric("Raw Rows", len(df_raw))
        col2.metric("Raw Columns", len(df_raw.columns))
    
    st.markdown("### API Serving")
    try:
        import requests
        res = requests.get("http://localhost:8000/health", timeout=2)
        if res.status_code == 200:
            st.success("API Online")
        else:
            st.warning(f"API Error: {res.status_code}")
    except:
        st.error("API Offline")
