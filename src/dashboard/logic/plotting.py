
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Plotly theme — Cấu hình chung cho tất cả biểu đồ Plotly (LIGHT MODE)
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',    # Transparent
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#24292e', family='Inter, -apple-system, BlinkMacSystemFont, sans-serif'), # GitHub Black
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='#e1e4e8',
        borderwidth=1,
        font=dict(color='#586069')
    ),
    xaxis=dict(
        gridcolor='#e1e4e8',
        linecolor='#e1e4e8',
        tickfont=dict(color='#586069')
    ),
    yaxis=dict(
        gridcolor='#e1e4e8',
        linecolor='#e1e4e8',
        tickfont=dict(color='#586069')
    )
)

# Bảng màu gradient cho biểu đồ (Light Mode Friendly)
COLORS = {
    'primary': '#0969da',      # GitHub Blue
    'secondary': '#8250df',    # GitHub Purple
    'success': '#1a7f37',      # GitHub Green
    'danger': '#cf222e',       # GitHub Red
    'warning': '#9a6700',      # GitHub Orange
    'info': '#218bff',         # GitHub Light Blue
    'gradient': ['#0969da', '#54aeff', '#80ccff', '#b3e1ff', '#ddf4ff'],
    'churn': ['#1a7f37', '#cf222e'],  # Green (Retained) / Red (Churned)
}

def create_churn_donut_chart(retained, churned, churn_rate):
    # Tạo biểu đồ Donut tỷ lệ churn.
    fig_donut = go.Figure(data=[go.Pie(
        labels=['Trung Thành', 'Rời Bỏ'],
        values=[retained, churned],
        hole=0.65,
        marker=dict(colors=COLORS['churn']),
        textinfo='percent+label',
        textfont=dict(size=14, color='white'),
        hovertemplate='%{label}: %{value:,} users<br>%{percent}<extra></extra>'
    )])
    fig_donut.update_layout(
        **PLOTLY_LAYOUT,
        height=350,
        showlegend=False,
        annotations=[dict(
            text=f"<b>{churn_rate:.0%}</b><br>Rời Bỏ",
            x=0.5, y=0.5, font_size=22, showarrow=False,
            font=dict(color='#e2e8f0')
        )]
    )
    return fig_donut

def create_age_distribution_chart(plot_df):
    # Tạo biểu đồ phân bố độ tuổi.
    fig_age = px.histogram(
        plot_df, x='age', color='Trạng Thái',
        nbins=30,
        color_discrete_map={'Trung Thành': COLORS['success'], 'Rời Bỏ': COLORS['danger']},
        labels={'age': 'Tuổi', 'Trạng Thái': 'Trạng Thái', 'count': 'Số lượng'},
        barmode='overlay',
        opacity=0.7
    )
    fig_age.update_layout(**PLOTLY_LAYOUT, height=350)
    fig_age.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
    fig_age.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
    return fig_age

def create_rfm_boxplot(plot_df, col_name, label_name):
    # Tạo biểu đồ boxplot cho RFM.
    fig_box = px.box(
        plot_df, x='Trạng Thái', y=col_name,
        color='Trạng Thái',
        color_discrete_map={'Trung Thành': COLORS['success'], 'Rời Bỏ': COLORS['danger']},
        labels={col_name: label_name, 'Trạng Thái': 'Trạng Thái'},
    )
    fig_box.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
    fig_box.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
    fig_box.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
    return fig_box

def create_country_churn_chart(country_churn):
    # Tạo biểu đồ churn theo quốc gia.
    fig_country = px.bar(
        country_churn, x='churn_rate', y='country',
        orientation='h',
        color='churn_rate',
        color_continuous_scale=['#10b981', '#f59e0b', '#ef4444'],
        labels={'churn_rate': 'Tỷ Lệ Rời Bỏ', 'country': 'Quốc Gia'}
    )
    fig_country.update_layout(**PLOTLY_LAYOUT, height=400)
    fig_country.update_xaxes(gridcolor='rgba(255,255,255,0.05)', tickformat='.0%')
    fig_country.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
    return fig_country

def create_behavior_scatter(sample_df, col_x, col_y):
    # Tạo biểu đồ scatter hành vi mua sắm.
    fig_scatter = px.scatter(
        sample_df, x=col_x, y=col_y, color='Trạng Thái',
        color_discrete_map={'Trung Thành': COLORS['success'], 'Rời Bỏ': COLORS['danger']},
        opacity=0.4,
        labels={'Trạng Thái': 'Trạng Thái'}
    )
    fig_scatter.update_layout(**PLOTLY_LAYOUT, height=500)
    fig_scatter.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
    fig_scatter.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
    return fig_scatter

def create_correlation_heatmap(corr_matrix):
    # Tạo heatmap tương quan.
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        labels=dict(color="Tương Quan")
    )
    fig_heatmap.update_layout(**PLOTLY_LAYOUT, height=500)
    return fig_heatmap

def create_feature_importance_chart(importance_df):
    # Tạo biểu đồ feature importance.
    fig_importance = px.bar(
        importance_df.sort_values('importance'),
        x='importance', y='feature',
        orientation='h',
        color='importance',
        color_continuous_scale=['#c4b5fd', '#6366f1'],
        labels={'importance': 'Mức Độ Ảnh Hưởng', 'feature': 'Yếu Tố'}
    )
    fig_importance.update_layout(**PLOTLY_LAYOUT, height=500, coloraxis_showscale=False)
    fig_importance.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
    fig_importance.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
    return fig_importance
