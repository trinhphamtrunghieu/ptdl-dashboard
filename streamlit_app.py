import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import os
import glob
import statsmodels.api as sm
from linearmodels import PanelOLS

# Cấu hình trang Streamlit
st.set_page_config(page_title="Phân tích Di cư Lao động", layout="wide")

# --- ĐỊNH NGHĨA TÊN CỘT TIẾNG VIỆT & MAPPING CHO BIẾN GÂY NHIỄU ---
COL_NAMES_MAP = {
    "hhid": "Mã hộ",
    "province": "Tỉnh",
    "year_std": "Năm",
    "migrant": "Có di cư (1=Có, 0=Không)",
    "wmigr": "Di cư vì việc làm (1=Có, 0=Không)",
    "other_migrant": "Di cư lý do khác (1=Có, 0=Không)",
    "dremit": "Nhận kiều hối (1=Có, 0=Không)",
    "dremit2": "Nhận kiều hối từ nguồn khác (1=Có, 0=Không)",
    "quintile": "Phân vị thu nhập (Nhóm 1-5)",
    "natshock_bin": "Cú sốc thiên tai (1=Có, 0=Không)",
    "econshock_bin": "Cú sốc kinh tế (1=Có, 0=Không)",
    "rhhincome": "Thu nhập thực (Nghìn VNĐ)",
    "age": "Tuổi chủ hộ",
    "totareaown": "Diện tích đất (m2)",
    "femalehead_bin": "Chủ hộ nữ (1=Có, 0=Không)",
    "kinh": "Dân tộc Kinh (1=Có, 0=Không)",
    "dfoodexp_pc": "Mức thay đổi chi tiêu thực phẩm/người (Nghìn VNĐ)",
    "damtbor": "Mức thay đổi số tiền đi vay (Nghìn VNĐ)",
    "income_asinh": "Thu nhập thực (arcsinh)"
}

CONFOUNDER_MAP = {
    "Tuổi chủ hộ": "age",
    "Diện tích đất (m2)": "totareaown",
    "Chủ hộ nữ": "femalehead_bin",
    "Dân tộc Kinh": "kinh",
    "Cú sốc thiên tai": "natshock_bin",
    "Cú sốc kinh tế": "econshock_bin"
}

# Hàm chuẩn hóa năm
def std_year(y):
    if pd.isna(y):
        return np.nan
    return 2000 + y if y < 100 else y

# Hàm chuyển đổi nhị phân
def to_binary(s):
    return (s.astype(str).str.strip().str.lower() == "yes").astype(int)

# --- HÀM TÌM KIẾM FILE TRONG THƯ MỤC ---
def find_data_file(base_dir="."):
    dta_files = glob.glob(os.path.join(base_dir, "**", "*7a*.dta"), recursive=True)
    if dta_files:
        return dta_files[0]
    csv_files = glob.glob(os.path.join(base_dir, "**", "varhs_combined_data.csv"), recursive=True)
    if csv_files:
        return csv_files[0]
    return None

# --- CACHE DỮ LIỆU ĐỂ TỐI ƯU HIỆU NĂNG ---
@st.cache_data
def load_and_clean_data(file_path):
    if file_path.endswith('.dta'):
        c7a = pd.read_stata(file_path)
    else:
        raw = pd.read_csv(file_path, low_memory=False)
        c7a = raw[raw["source_file"] == "Chapter_7a.dta"].copy()

    c7a["year_std"] = c7a["year"].apply(std_year)
    c7a["femalehead_bin"] = to_binary(c7a["femalehead"])
    c7a["natshock_bin"] = to_binary(c7a["natshock"])
    c7a["econshock_bin"] = to_binary(c7a["econshock"])
    c7a["hhid"] = (
        c7a["province"].astype(str) + "_" + c7a["district"].astype(str) + "_" +
        c7a["commune"].astype(str) + "_" + c7a["household"].astype(str)
    )

    keep_cols = [
        "hhid", "province", "year_std",
        "migrant", "wmigr", "other_migrant", "dremit", "dremit2", "quintile",
        "natshock_bin", "econshock_bin", "rhhincome", "age", "totareaown",
        "femalehead_bin", "kinh", "dfoodexp_pc", "damtbor",
    ]
    
    panel = c7a[keep_cols].drop_duplicates(subset=["hhid", "year_std"]).copy()
    panel = panel.dropna(subset=["migrant", "age", "totareaown", "femalehead_bin", "kinh"])
    panel["income_asinh"] = np.arcsinh(panel["rhhincome"])
    return panel

# --- GIAO DIỆN CHÍNH ---
st.title("Tác động của Di cư Lao động đến Chi tiêu & An sinh Hộ gia đình")
st.markdown("""
**Bộ dữ liệu:** Vietnam Access to Resources Household Survey (VARHS), UNU-WIDER  
**Câu hỏi nghiên cứu:** Việc hộ gia đình có thành viên di cư lao động tác động như thế nào đến chi tiêu và an sinh tài chính của hộ?
""")

data_file_path = find_data_file("./")

if data_file_path is not None:
    try:
        panel = load_and_clean_data(data_file_path)
        
        # --- SIDEBAR TƯƠNG TÁC BIẾN GÂY NHIỄU ---
        st.sidebar.header("Cấu hình Mô hình")
        st.sidebar.markdown("Chọn các **biến gây nhiễu (Confounders)** đưa vào phân tích:")
        
        selected_confounders_ui = st.sidebar.multiselect(
            "Biến quan sát được:",
            options=list(CONFOUNDER_MAP.keys()),
            default=list(CONFOUNDER_MAP.keys()) # Mặc định chọn tất cả
        )
        
        # Ánh xạ từ tên tiếng Việt sang tên cột trong dataframe
        selected_covars = [CONFOUNDER_MAP[c] for c in selected_confounders_ui]
        
        # 5 Tab chính
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dữ liệu Panel", 
            "📈 Cân bằng", 
            "🕸️ DAG", 
            "⚖️ Mô hình 1: PSM/IPTW", 
            "⏳ Mô hình 2: Panel Fixed-Effects"
        ])
        
        # --- TAB 1: HIỂN THỊ DỮ LIỆU CÓ PHÂN TRANG ---
        with tab1:
            st.subheader("Bộ dữ liệu phân tích (Panel hộ gia đình 2012–2014)")
            st.markdown(f"**Tổng số dòng:** {panel.shape[0]} | **Số cột:** {panel.shape[1]}")
            
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                rows_per_page = st.selectbox("Số dòng mỗi trang:", options=list(range(50, 110, 10)), index=0)
            total_pages = (len(panel) - 1) // rows_per_page + 1 
            with col2:
                page_num = st.selectbox("Chọn trang:", options=list(range(1, total_pages + 1)), index=0)
            
            start_idx = (page_num - 1) * rows_per_page
            end_idx = start_idx + rows_per_page
            paginated_df = panel.iloc[start_idx:end_idx]
            
            st.dataframe(paginated_df.rename(columns=COL_NAMES_MAP), use_container_width=True)
            st.caption(f"Đang hiển thị từ dòng **{start_idx + 1}** đến **{min(end_idx, len(panel))}** trên tổng số **{len(panel)}** dòng.")
            
            st.markdown("---")
            st.markdown("### Tỷ lệ hộ có di cư lao động theo năm")
            df_migrant_rate = panel.groupby("year_std")["migrant"].mean().round(3).reset_index()
            st.dataframe(df_migrant_rate.rename(columns=COL_NAMES_MAP))

        # --- TAB 2: THỐNG KÊ & BẢNG CÂN BẰNG ---
        with tab2:
            st.subheader("Kiểm tra cân bằng (Balance check)")
            st.markdown("Nhóm hộ **có di cư** và **không di cư** có sự khác biệt hệ thống về các đặc điểm nền (Bias).")
            
            # Cập nhật danh sách hiển thị dựa trên biến đã chọn (mặc định lấy theo list đầy đủ nếu không bị bỏ check)
            covars_to_show = selected_covars if selected_covars else list(CONFOUNDER_MAP.values())
            outcomes = ["dfoodexp_pc", "damtbor", "income_asinh"]
            
            balance = panel.groupby("migrant")[covars_to_show + outcomes].mean().T
            balance.columns = ["Không di cư", "Có di cư"]
            balance["Chênh lệch"] = balance["Có di cư"] - balance["Không di cư"]
            
            balance.index = [COL_NAMES_MAP.get(idx, idx) for idx in balance.index]
            st.dataframe(balance.round(3))
            
            st.subheader("Biểu đồ mô tả")
            sns.set_theme(style="whitegrid")
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

            sns.barplot(data=panel, x="migrant", y="dfoodexp_pc", ax=axes[0], palette="Blues")
            axes[0].set_title("Mức thay đổi chi tiêu thực phẩm/người")
            axes[0].set_xlabel("0 = Không | 1 = Có di cư")

            sns.barplot(data=panel, x="migrant", y="damtbor", ax=axes[1], palette="Oranges")
            axes[1].set_title("Mức thay đổi số tiền đi vay")
            axes[1].set_xlabel("0 = Không | 1 = Có di cư")

            mig_rate = panel.groupby("year_std")["migrant"].mean()
            axes[2].plot(mig_rate.index, mig_rate.values, marker="o", linewidth=2)
            axes[2].set_title("Tỷ lệ hộ có di cư lao động theo năm")
            axes[2].set_ylim(0, 0.4)
            axes[2].set_xticks(mig_rate.index)

            plt.tight_layout()
            st.pyplot(fig) 

        # --- TAB 3: ĐỒ THỊ DAG (GRAPHVIZ) ---
        with tab3:
            st.subheader("Đồ thị nhân quả (DAG)")
            st.markdown("Biểu diễn giả định nhân quả. Các nút màu xanh dương là các biến gây nhiễu đang được kiểm soát.")
            
            # Tạo đồ thị bằng Graphviz
            graph = graphviz.Digraph()
            graph.attr(rankdir='LR', size='10,6') # Từ trái sang phải
            with graph.subgraph(name='cluster_legend') as c:
                c.attr(label='Legend', color='black', style='solid', rank='sink')
                
                # Define legend items as key-value pairs of nodes
                c.node('key1', 'Biến gây nhiễu (cofounder)', fillcolor='#a6cee3', style='filled')
                c.node('key2', 'Biến can thiệp (treatment)', fillcolor='#fb9a99', style='filled')
                c.node('key3', 'Biến trung gian (mediator)', fillcolor='#fdbf6f', style='filled')
                c.node('key4', 'Biến kết quả (outcome)', fillcolor='#b2df8a', style='filled')
            graph.render('graph_with_legend', format='png', cleanup=True)
            # Bảng màu
            color_treatment = "#fb9a99"
            color_mediator = "#fdbf6f"
            color_outcome = "#b2df8a"
            color_confounder = "#a6cee3"

            # 1. Các Node cốt lõi
            graph.node("Di cư lao động", style="filled", fillcolor=color_treatment, shape="box")
            graph.node("Nhận kiều hối", style="filled", fillcolor=color_mediator, shape="box")
            
            outcomes_dag = ["Thay đổi chi tiêu thực phẩm", "Thay đổi vay mượn", "Thu nhập hộ"]
            for o in outcomes_dag:
                graph.node(o, style="filled", fillcolor=color_outcome, shape="ellipse")
                
            # 2. Định nghĩa các cung kết nối nhân quả chính
            graph.edge("Di cư lao động", "Nhận kiều hối")
            for o in outcomes_dag:
                graph.edge("Di cư lao động", o)
                graph.edge("Nhận kiều hối", o)
                
            # 3. Kết nối Biến gây nhiễu (Dựa trên Sidebar MultiSelect)
            if len(selected_confounders_ui) > 0:
                for c in selected_confounders_ui:
                    graph.node(c, style="filled", fillcolor=color_confounder, shape="ellipse")
                    graph.edge(c, "Di cư lao động")
                    for o in outcomes_dag:
                        graph.edge(c, o)
            else:
                st.warning("⚠️ Không có biến gây nhiễu nào được chọn.")

            st.graphviz_chart(graph)

        # --- TAB 4: MÔ HÌNH 1 (IPTW / PSM) ---
        with tab4:
            st.subheader("Mô hình 1: Inverse Probability of Treatment Weighting (IPTW)")
            st.markdown("Kiểm soát thiên lệch chọn mẫu dựa trên các đặc điểm quan sát được.")
            
            if len(selected_covars) > 0:
                # Nếu có chọn biến gây nhiễu -> Tính Propensity Score
                X = sm.add_constant(panel[selected_covars])
                ps_model = sm.Logit(panel["migrant"], X).fit(disp=0)
                ps = ps_model.predict(X)
                weights = np.where(panel["migrant"] == 1, 1/ps, 1/(1-ps))
                st.info(f"✅ Đã tính toán thành công Trọng số dựa trên **{len(selected_covars)}** biến gây nhiễu. Min weight: {weights.min():.2f}, Max weight: {weights.max():.2f}")
            else:
                # Nếu không chọn biến gây nhiễu nào -> So sánh trung bình thô (Naive estimate)
                weights = np.ones(len(panel))
                st.warning("⚠️ Đang phân tích tác động thô (Mặc định Trọng số = 1) do không có biến gây nhiễu nào được kiểm soát.")
            
            results_iptw = []
            for outcome in outcomes:
                valid_idx = panel[outcome].notna()
                
                y_valid = panel.loc[valid_idx, outcome]
                X_valid = sm.add_constant(panel.loc[valid_idx, "migrant"])
                w_valid = weights[valid_idx]
                
                wls_model = sm.WLS(y_valid, X_valid, weights=w_valid).fit()
                
                results_iptw.append({
                    "Biến Kết Quả": COL_NAMES_MAP.get(outcome, outcome),
                    "Hiệu ứng ATE (Hệ số)": wls_model.params["migrant"],
                    "Sai số chuẩn (Std.Err)": wls_model.bse["migrant"],
                    "p-value": wls_model.pvalues["migrant"],
                    "Ý nghĩa": "Có (p < 0.05)" if wls_model.pvalues["migrant"] < 0.05 else "Không"
                })
            
            st.dataframe(pd.DataFrame(results_iptw).style.format({"Hiệu ứng ATE (Hệ số)": "{:.3f}", "Sai số chuẩn (Std.Err)": "{:.3f}", "p-value": "{:.4f}"}))

        # --- TAB 5: MÔ HÌNH 2 (PANEL FIXED-EFFECTS) ---
        with tab5:
            st.subheader("Mô hình 2: Panel Fixed-Effects (FE)")
            st.markdown("Khai thác dữ liệu bảng để loại bỏ biến gây nhiễu không đổi theo thời gian.")
            
            panel_fe = panel.set_index(["hhid", "year_std"])
            
            # Cập nhật danh sách biến đưa vào FE dựa trên thanh điều khiển
            covars_fe = ["migrant"] + selected_covars
            
            results_fe = []
            for outcome in outcomes:
                valid_idx = panel_fe[outcome].notna()
                
                y_valid = panel_fe.loc[valid_idx, outcome]
                exog_valid = sm.add_constant(panel_fe.loc[valid_idx, covars_fe])
                
                # drop_absorbed=True sẽ tự động lược bỏ các biến không đổi theo thời gian (giới tính, tuổi, dân tộc...)
                fe_model = PanelOLS(y_valid, exog_valid, entity_effects=True, drop_absorbed=True).fit()
                
                results_fe.append({
                    "Biến Kết Quả": COL_NAMES_MAP.get(outcome, outcome),
                    "Hiệu ứng (Hệ số)": fe_model.params["migrant"],
                    "Sai số chuẩn (Std.Err)": fe_model.std_errors["migrant"],
                    "p-value": fe_model.pvalues["migrant"],
                    "Ý nghĩa": "Có (p < 0.05)" if fe_model.pvalues["migrant"] < 0.05 else "Không"
                })
                
            st.dataframe(pd.DataFrame(results_fe).style.format({"Hiệu ứng (Hệ số)": "{:.3f}", "Sai số chuẩn (Std.Err)": "{:.3f}", "p-value": "{:.4f}"}))
            
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi xử lý dữ liệu: {e}")

else:
    st.error("❌ Không tìm thấy file dữ liệu.\n")