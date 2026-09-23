import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="AI Supply Chain Dashboard", layout="wide")
st.title("📊 HỆ THỐNG QUẢN TRỊ TỒN KHO: GIAO THOA TÀI CHÍNH & AI")

def load_data():
    df_fin = pd.read_csv('financial_metadata_100.csv')
    df_fcst = pd.read_csv('ai_forecast_output_100.csv')
    df_fcst['ds'] = pd.to_datetime(df_fcst['ds'])

    df_hist = pd.read_csv('historical_sales_100.csv')
    df_hist['ds'] = pd.to_datetime(df_hist['ds'])

    df_fin['Cu'] = df_fin['Price'] - df_fin['Cost']
    df_fin['Co'] = df_fin['Cost'] - df_fin['Salvage']
    df_fin['Optimal_q'] = df_fin['Cu'] / (df_fin['Cu'] + df_fin['Co'])

    df_master = pd.merge(df_fin, df_fcst, on='SKU')
    return df_master, df_hist


df_master, df_hist = load_data()

# =====================================================================
# BẢNG ĐIỀU KHIỂN (SIDEBAR) - SEARCH TƯƠNG ĐỐI & INTERACTIVE TABLE
# =====================================================================
st.sidebar.header("⚙️ Bảng Điều Khiển")

# 1. Trích xuất bảng thông tin duy nhất cho Sidebar
df_unique_sku = df_master[['SKU', 'Price', 'Cost', 'Salvage']].drop_duplicates().reset_index(drop=True)

if 'selected_sku' not in st.session_state:
    st.session_state.selected_sku = df_unique_sku['SKU'].iloc[0]

# 2. Thanh tìm kiếm tương đối (Search Bar)
search_kw = st.sidebar.text_input("🔍 Tìm kiếm SKU:", placeholder="Nhập tên mã (VD: 005, cable...)").strip()
if search_kw:
    # Lọc chuỗi tương đối, không phân biệt hoa/thường (case=False)
    df_unique_sku = df_unique_sku[df_unique_sku['SKU'].str.contains(search_kw, case=False, na=False)]

st.sidebar.markdown("**Danh sách Sản Phẩm (Click vào dòng để xem biểu đồ):**")

# 3. Phân loại theo tiền tố
groups = {
    "📱 Điện thoại (TECH)": df_unique_sku[df_unique_sku['SKU'].str.contains("TECH")],
    "🔌 Phụ kiện (ACC)": df_unique_sku[df_unique_sku['SKU'].str.contains("ACC")],
    "🍔 Thực phẩm (FMCG)": df_unique_sku[df_unique_sku['SKU'].str.contains("FMCG")]
}

for group_name, df_group in groups.items():
    if df_group.empty:
        continue

    df_group = df_group.reset_index(drop=True)
    is_expanded = st.session_state.selected_sku in df_group['SKU'].values

    with st.sidebar.expander(f"{group_name} ({len(df_group)})", expanded=is_expanded):
        # 4. Giao diện Table tương tác
        event = st.dataframe(
            df_group,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key=f"grid_{group_name}"
        )

        # Bắt sự kiện khi người dùng click vào một dòng trong Table
        if event.selection.rows:
            selected_idx = event.selection.rows[0]
            clicked_sku = df_group.iloc[selected_idx]['SKU']
            if st.session_state.selected_sku != clicked_sku:
                st.session_state.selected_sku = clicked_sku
                st.rerun()

selected_sku = st.session_state.selected_sku

# =====================================================================
# TRÍCH XUẤT VÀ TÍNH TOÁN DỮ LIỆU
# =====================================================================
sku_data = df_master[df_master['SKU'] == selected_sku].sort_values('ds')
hist_data = df_hist[df_hist['unique_id'] == selected_sku].tail(30).sort_values('ds')

st.subheader("1. 🛠️ Trình Mô Phỏng Chi Phí (Newsvendor Model)")
col1, col2, col3 = st.columns(3)
first_row = sku_data.iloc[0]

with col1: sim_price = st.number_input("Giá Bán ($p$)", value=int(first_row['Price']), step=100)
with col2: sim_cost = st.number_input("Giá Vốn ($c$)", value=int(first_row['Cost']), step=100)
with col3: sim_salvage = st.number_input("Giá Thanh Lý ($s$)", value=int(first_row['Salvage']), step=100)

sim_Cu = sim_price - sim_cost
sim_Co = sim_cost - sim_salvage
sim_q = sim_Cu / (sim_Cu + sim_Co) if (sim_Cu + sim_Co) > 0 else 0.5

st.info(f"**Tỷ lệ tới hạn (Critical Ratio) mục tiêu:** $q^* = {sim_q:.2f}$")

available_quantiles = [0.10, 0.30, 0.50, 0.70, 0.90]
closest_q = min(available_quantiles, key=lambda x: abs(x - sim_q))
quantile_columns = {0.10: 'P10', 0.30: 'P30', 0.50: 'P50', 0.70: 'P70', 0.90: 'P90'}
target_col = quantile_columns[closest_q]
next_day_order = sku_data.iloc[0][target_col]

c1, c2, c3 = st.columns(3)
if closest_q >= 0.7:
    strat_label = "TẤN CÔNG 🚀"
elif closest_q <= 0.3:
    strat_label = "PHÒNG THỦ 🛡️"
else:
    strat_label = "CÂN BẰNG ⚖️"

c1.metric(label="Chiến Lược Gán Nhãn", value=strat_label)
c2.metric(label="Phân Vị Khớp Lệnh", value=f"P{int(closest_q * 100)}")
c3.metric(label="Lệnh Nhập Hàng (T+1)", value=f"{next_day_order} Units")

# =====================================================================
# TRỰC QUAN HÓA BẰNG PLOTLY
# =====================================================================
st.subheader("2. 📈 Hình Thái Học Khoảng Dự Báo (Prediction Intervals)")
fig = go.Figure()

fig.add_trace(go.Scatter(x=hist_data['ds'], y=hist_data['y'], mode='lines+markers', name='Thực tế bán hàng',
                         line=dict(color='black', width=2)))

last_hist_date = hist_data['ds'].iloc[-1]
last_hist_val = hist_data['y'].iloc[-1]

future_dates = pd.concat([pd.Series([last_hist_date]), sku_data['ds']])
f_p10 = pd.concat([pd.Series([last_hist_val]), sku_data['P10']])
f_p50 = pd.concat([pd.Series([last_hist_val]), sku_data['P50']])
f_p90 = pd.concat([pd.Series([last_hist_val]), sku_data['P90']])
f_target = pd.concat([pd.Series([last_hist_val]), sku_data[target_col]])

fig.add_trace(go.Scatter(x=future_dates, y=f_p90, mode='lines', line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=future_dates, y=f_p10, mode='lines', fill='tonexty', fillcolor='rgba(135, 206, 250, 0.3)',
                         line=dict(width=0), name='Dải băng PI (80%)'))

fig.add_trace(
    go.Scatter(x=future_dates, y=f_p50, mode='lines', line=dict(color='blue', dash='dash'), name='Kỳ vọng AI (P50)'))
fig.add_trace(go.Scatter(x=future_dates, y=f_target, mode='lines+markers', line=dict(color='red', width=3, dash='dot'),
                         name=f'Quyết định vận hành (P{int(closest_q * 100)})'))

last_date = hist_data['ds'].max()
fig.add_vline(x=last_date, line_width=2, line_dash="dash", line_color="red")
fig.update_layout(title=f'Trực quan hóa Khối lượng Nhập hàng Động cho {selected_sku}', xaxis_title='Thời gian',
                  yaxis_title='Số lượng (Units)', hovermode="x unified")

st.plotly_chart(fig, width='stretch')

# =====================================================================
# XUẤT BÁO CÁO EXCEL
# =====================================================================
st.markdown("### 📥 Xuất dữ liệu dự báo")

# Tạo bảng dữ liệu xuất Excel cho SKU đang được chọn
df_export = sku_data[
    ['SKU', 'ds', 'P10', 'P30', 'P50', 'P70', 'P90']
].copy()

df_export['Critical_Ratio'] = sim_q
df_export['Selected_Quantile'] = f"P{int(closest_q * 100)}"
df_export['Strategy'] = strat_label
df_export['Recommended_Order'] = df_export[target_col]

# Sắp xếp lại thứ tự cột
df_export = df_export[
    [
        'SKU',
        'ds',
        'P10',
        'P30',
        'P50',
        'P70',
        'P90',
        'Critical_Ratio',
        'Selected_Quantile',
        'Strategy',
        'Recommended_Order'
    ]
]

# Tạo file Excel trong bộ nhớ
excel_buffer = BytesIO()

with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
    df_export.to_excel(
        writer,
        index=False,
        sheet_name='Forecast'
    )

    # Thêm sheet thông số tài chính
    df_financial_export = pd.DataFrame({
        'SKU': [selected_sku],
        'Price': [sim_price],
        'Cost': [sim_cost],
        'Salvage': [sim_salvage],
        'Cu': [sim_Cu],
        'Co': [sim_Co],
        'Critical_Ratio': [sim_q],
        'Selected_Quantile': [f"P{int(closest_q * 100)}"],
        'Strategy': [strat_label],
        'Next_Day_Order': [next_day_order]
    })

    df_financial_export.to_excel(
        writer,
        index=False,
        sheet_name='Decision_Summary'
    )

excel_buffer.seek(0)

st.download_button(
    label="📥 XUẤT FILE EXCEL",
    data=excel_buffer.getvalue(),
    file_name=f"Bao_Cao_Du_Bao_{selected_sku}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    width="stretch"
)
