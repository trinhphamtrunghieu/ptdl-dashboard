import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
import glob
import statsmodels.api as sm
from linearmodels import PanelOLS

# Cấu hình trang Streamlit
st.set_page_config(page_title="Phân tích Di cư Lao động", layout="wide")

# --- ĐỊNH NGHĨA TÊN CỘT TIẾNG VIỆT CHO GIAO DIỆN (UI) ---
COL_NAMES_MAP = {
    "hhid": "Mã hộ",
    "province": "Tỉnh",
    "year_std": "Năm",
    "migrant": "Có di cư (1=Có, 0=Không)",
    "wmigr": "Di cư vì việc làm",
    "other_migrant": "Di cư lý do khác",
    "dremit": "Nhận kiều hối",
    "dremit2": "Kiều hối 2",
    "quintile": "Nhóm thu nhập",
    "natshock_bin": "Cú sốc thiên tai",
    "econshock_bin": "Cú sốc kinh tế",
    "rhhincome": "Thu nhập thực",
    "age": "Tuổi chủ hộ",
    "totareaown": "Diện tích đất",
    "femalehead_bin": "Chủ hộ nữ",
    "kinh": "Dân tộc Kinh",
    "dfoodexp_pc": "Mức thay đổi chi tiêu thực phẩm/người", # Đã sửa cho rõ nghĩa
    "damtbor": "Mức thay đổi số tiền đi vay", # Đã sửa cho rõ nghĩa
    "income_asinh": "Thu nhập (arcsinh)"
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
            
            # Phân chia cột cho các Dropdown
            col1, col2, col3 = st.columns([1, 1, 4])
            
            with col1:
                # Dropdown chọn số item mỗi trang (Từ 50 -> 100, bước nhảy 10)
                rows_per_page = st.selectbox(
                    "Số dòng mỗi trang:", 
                    options=list(range(50, 110, 10)), 
                    index=0
                )
            
            # Tính tổng số trang
            total_pages = (len(panel) - 1) // rows_per_page + 1 
            
            with col2:
                # Dropdown chọn trang
                page_num = st.selectbox(
                    "Chọn trang:", 
                    options=list(range(1, total_pages + 1)), 
                    index=0
                )
            
            # Cắt dữ liệu (slice) theo phân trang
            start_idx = (page_num - 1) * rows_per_page
            end_idx = start_idx + rows_per_page
            paginated_df = panel.iloc[start_idx:end_idx]
            
            # Hiển thị bảng
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
            
            covariates = ["age", "totareaown", "femalehead_bin", "kinh", "natshock_bin", "econshock_bin"]
            outcomes = ["dfoodexp_pc", "damtbor", "income_asinh"]
            
            balance = panel.groupby("migrant")[covariates + outcomes].mean().T
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

        # --- TAB 3: ĐỒ THỊ DAG ---
        with tab3:
            st.subheader("Đồ thị nhân quả (DAG)")
            
            G = nx.DiGraph()
            confounders = ["Tuổi chủ hộ", "Giới tính chủ hộ", "Dân tộc", "Diện tích đất",
                           "Tỉnh/vùng", "Cú sốc thiên tai", "Cú sốc kinh tế"]
            treatment = "Di cư lao động"
            mediator = "Nhận kiều hối"
            outcomes_dag = ["Thay đổi chi tiêu thực phẩm", "Thay đổi vay mượn / An sinh", "Thu nhập hộ"]

            for c in confounders:
                G.add_edge(c, treatment)
                for o in outcomes_dag:
                    G.add_edge(c, o)

            G.add_edge(treatment, mediator)
            for o in outcomes_dag:
                G.add_edge(treatment, o)
                G.add_edge(mediator, o)

            node_roles = {**{c: "confounder" for c in confounders}, treatment: "treatment", mediator: "mediator", **{o: "outcome" for o in outcomes_dag}}
            color_map = {"confounder": "#a6cee3", "treatment": "#fb9a99", "mediator": "#fdbf6f", "outcome": "#b2df8a"}
            node_colors = [color_map[node_roles[n]] for n in G.nodes()]

            pos = {
                "Tuổi chủ hộ": (-2, 3), "Giới tính chủ hộ": (-2, 2), "Dân tộc": (-2, 1),
                "Diện tích đất": (-2, 0), "Tỉnh/vùng": (-2, -1),
                "Cú sốc thiên tai": (-2, -2), "Cú sốc kinh tế": (-2, -3),
                "Di cư lao động": (0, 0), "Nhận kiều hối": (2, 0),
                "Thay đổi chi tiêu thực phẩm": (4, 1.5), "Thay đổi vay mượn / An sinh": (4, 0), "Thu nhập hộ": (4, -1.5),
            }

            fig_dag, ax_dag = plt.subplots(figsize=(12, 7))
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2600, edgecolors="black", ax=ax_dag)
            nx.draw_networkx_labels(G, pos, font_size=9, ax=ax_dag)
            nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=18, connectionstyle="arc3,rad=0.05", ax=ax_dag)
            ax_dag.axis("off")
            st.pyplot(fig_dag)

        # --- TAB 4: MÔ HÌNH 1 (IPTW / PSM) ---
        with tab4:
            st.subheader("Mô hình 1: Inverse Probability of Treatment Weighting (IPTW)")
            st.markdown("Kiểm soát thiên lệch chọn mẫu dựa trên các đặc điểm quan sát được (Selection on Observables).")
            
            covars_iptw = ["age", "totareaown", "femalehead_bin", "kinh", "natshock_bin", "econshock_bin"]
            X = sm.add_constant(panel[covars_iptw])
            ps_model = sm.Logit(panel["migrant"], X).fit(disp=0)
            ps = ps_model.predict(X)
            
            weights = np.where(panel["migrant"] == 1, 1/ps, 1/(1-ps))
            
            st.info(f"Đã tính toán thành công Trọng số (Weights). Min weight: {weights.min():.2f}, Max weight: {weights.max():.2f}")
            
            results_iptw = []
            for outcome in outcomes:
                X_out = sm.add_constant(panel["migrant"])
                wls_model = sm.WLS(panel[outcome], X_out, weights=weights).fit()
                
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
            st.markdown("Sử dụng kỹ thuật Fixed-Effects theo Hộ gia đình để loại bỏ các biến gây nhiễu không đổi theo thời gian (Unobserved Time-Invariant Heterogeneity).")
            
            panel_fe = panel.set_index(["hhid", "year_std"])
            covars_fe = ["migrant", "natshock_bin", "econshock_bin"]
            
            results_fe = []
            for outcome in outcomes:
                exog = sm.add_constant(panel_fe[covars_fe])
                fe_model = PanelOLS(panel_fe[outcome], exog, entity_effects=True, drop_absorbed=True).fit()
                
                results_fe.append({
                    "Biến Kết Quả": COL_NAMES_MAP.get(outcome, outcome),
                    "Hiệu ứng (Hệ số)": fe_model.params["migrant"],
                    "Sai số chuẩn (Std.Err)": fe_model.std_errors["migrant"],
                    "p-value": fe_model.pvalues["migrant"],
                    "Ý nghĩa": "Có (p < 0.05)" if fe_model.pvalues["migrant"] < 0.05 else "Không"
                })
                
            st.dataframe(pd.DataFrame(results_fe).style.format({"Hiệu ứng (Hệ số)": "{:.3f}", "Sai số chuẩn (Std.Err)": "{:.3f}", "p-value": "{:.4f}"}))
            st.success("Cả 2 mô hình đã đưa ra các kết quả ước lượng giúp so sánh và phân tích nhân quả tốt hơn.")

    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi xử lý dữ liệu: {e}")

else:
    st.error("❌ Không tìm thấy file dữ liệu.")