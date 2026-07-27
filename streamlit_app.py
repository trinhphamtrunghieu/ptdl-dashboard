import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os

# Cấu hình trang Streamlit
st.set_page_config(page_title="Phân tích Di cư Lao động", layout="wide")

# Hàm chuẩn hóa năm
def std_year(y):
    if pd.isna(y):
        return np.nan
    return 2000 + y if y < 100 else y

# Hàm chuyển đổi nhị phân
def to_binary(s):
    return (s.astype(str).str.strip().str.lower() == "yes").astype(int)

# --- CACHE DỮ LIỆU ĐỂ TỐI ƯU HIỆU NĂNG ---
@st.cache_data
def load_and_clean_data(uploaded_file):
    # Đọc dữ liệu
    if uploaded_file.name.endswith('.dta'):
        c7a = pd.read_stata(uploaded_file)
    else:
        raw = pd.read_csv(uploaded_file, low_memory=False)
        c7a = raw[raw["source_file"] == "Chapter_7a.dta"].copy()

    # Chuẩn hóa năm
    c7a["year_std"] = c7a["year"].apply(std_year)
    
    # Nhị phân hóa
    c7a["femalehead_bin"] = to_binary(c7a["femalehead"])
    c7a["natshock_bin"] = to_binary(c7a["natshock"])
    c7a["econshock_bin"] = to_binary(c7a["econshock"])

    # Tạo mã hộ định danh
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
st.title("Tác động của Di cư Lao động đến Chi tiêu & An sinh Hộ gia đình Nông thôn Việt Nam")
st.markdown("""
**Bộ dữ liệu:** Vietnam Access to Resources Household Survey (VARHS), UNU-WIDER  
**Câu hỏi nghiên cứu:** Việc hộ gia đình có thành viên di cư lao động tác động như thế nào đến chi tiêu và an sinh tài chính của hộ?
""")

# --- SIDEBAR: TẢI DỮ LIỆU ---
st.sidebar.header("Cấu hình Dữ liệu")
uploaded_file = st.sidebar.file_uploader("Tải lên file Chapter_7a.dta hoặc varhs_combined_data.csv", type=["dta", "csv"])

if uploaded_file is not None:
    try:
        # Load dữ liệu
        panel = load_and_clean_data(uploaded_file)
        st.sidebar.success("✅ Đã tải và làm sạch dữ liệu thành công!")
        
        # Tạo các tab để hiển thị nội dung
        tab1, tab2, tab3 = st.tabs(["📊 Dữ liệu Panel", "📈 Thống kê & Cân bằng", "🕸️ Khung nhân quả (DAG)"])
        
        # --- TAB 1: HIỂN THỊ DỮ LIỆU ---
        with tab1:
            st.subheader("Bộ dữ liệu phân tích (Panel hộ gia đình 2012–2014)")
            st.markdown(f"**Số dòng sau làm sạch:** {panel.shape[0]} | **Số cột:** {panel.shape[1]}")
            st.dataframe(panel.head(50)) # Hiển thị 50 dòng đầu cho nhẹ giao diện
            
            st.markdown("### Tỷ lệ hộ có di cư lao động theo năm")
            st.dataframe(panel.groupby("year_std")["migrant"].mean().round(3).reset_index())

        # --- TAB 2: THỐNG KÊ & BẢNG CÂN BẰNG ---
        with tab2:
            st.subheader("Kiểm tra cân bằng (Balance check) giữa các nhóm")
            st.markdown("Nhóm hộ **có di cư** và **không di cư** có sự khác biệt hệ thống về các đặc điểm nền.")
            
            covariates = ["age", "totareaown", "femalehead_bin", "kinh", "natshock_bin", "econshock_bin"]
            outcomes = ["dfoodexp_pc", "damtbor", "income_asinh"]
            
            balance = (
                panel.groupby("migrant")[covariates + outcomes]
                .mean()
                .T
                .rename(columns={0: "Không di cư", 1: "Có di cư"})
            )
            balance["Chênh lệch"] = balance["Có di cư"] - balance["Không di cư"]
            st.dataframe(balance.round(3))
            
            st.subheader("Biểu đồ mô tả")
            sns.set_theme(style="whitegrid")
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

            sns.barplot(data=panel, x="migrant", y="dfoodexp_pc", ax=axes[0], palette="Blues")
            axes[0].set_title("Thay đổi chi tiêu thực phẩm/người\\n(theo tình trạng di cư)")
            axes[0].set_xlabel("0 = Không di cư | 1 = Có di cư")

            sns.barplot(data=panel, x="migrant", y="damtbor", ax=axes[1], palette="Oranges")
            axes[1].set_title("Thay đổi số tiền vay mượn\\n(theo tình trạng di cư)")
            axes[1].set_xlabel("0 = Không di cư | 1 = Có di cư")

            mig_rate = panel.groupby("year_std")["migrant"].mean()
            axes[2].plot(mig_rate.index, mig_rate.values, marker="o", linewidth=2)
            axes[2].set_title("Tỷ lệ hộ có di cư lao động theo năm")
            axes[2].set_ylim(0, 0.4)
            axes[2].set_xticks(mig_rate.index)

            plt.tight_layout()
            st.pyplot(fig) # Hiển thị biểu đồ lên Streamlit

        # --- TAB 3: ĐỒ THỊ DAG ---
        with tab3:
            st.subheader("Đồ thị nhân quả (DAG)")
            st.markdown("Biểu diễn giả định về cấu trúc nhân quả giữa các biến trong mô hình.")
            
            G = nx.DiGraph()

            confounders = ["Tuổi chủ hộ", "Giới tính chủ hộ", "Dân tộc", "Diện tích đất",
                           "Tỉnh/vùng", "Cú sốc thiên tai", "Cú sốc kinh tế"]
            treatment = "Di cư lao động"
            mediator = "Nhận kiều hối"
            outcomes = ["Chi tiêu thực phẩm", "Vay mượn / An sinh", "Thu nhập hộ"]

            for c in confounders:
                G.add_edge(c, treatment)
                for o in outcomes:
                    G.add_edge(c, o)

            G.add_edge(treatment, mediator)
            for o in outcomes:
                G.add_edge(treatment, o)
                G.add_edge(mediator, o)

            node_roles = {**{c: "confounder" for c in confounders},
                          treatment: "treatment",
                          mediator: "mediator",
                          **{o: "outcome" for o in outcomes}}

            color_map = {"confounder": "#a6cee3", "treatment": "#fb9a99",
                         "mediator": "#fdbf6f", "outcome": "#b2df8a"}
            node_colors = [color_map[node_roles[n]] for n in G.nodes()]

            pos = {
                "Tuổi chủ hộ": (-2, 3), "Giới tính chủ hộ": (-2, 2), "Dân tộc": (-2, 1),
                "Diện tích đất": (-2, 0), "Tỉnh/vùng": (-2, -1),
                "Cú sốc thiên tai": (-2, -2), "Cú sốc kinh tế": (-2, -3),
                "Di cư lao động": (0, 0),
                "Nhận kiều hối": (2, 0),
                "Chi tiêu thực phẩm": (4, 1.5),
                "Vay mượn / An sinh": (4, 0),
                "Thu nhập hộ": (4, -1.5),
            }

            fig_dag, ax_dag = plt.subplots(figsize=(13, 8))
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2600, edgecolors="black", ax=ax_dag)
            nx.draw_networkx_labels(G, pos, font_size=9, ax=ax_dag)
            nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=18, connectionstyle="arc3,rad=0.05", ax=ax_dag)

            from matplotlib.patches import Patch
            legend_elems = [
                Patch(facecolor=color_map["confounder"], edgecolor="black", label="Biến gây nhiễu (confounder)"),
                Patch(facecolor=color_map["treatment"], edgecolor="black", label="Biến can thiệp (treatment)"),
                Patch(facecolor=color_map["mediator"], edgecolor="black", label="Biến trung gian (mediator)"),
                Patch(facecolor=color_map["outcome"], edgecolor="black", label="Kết quả (outcome)")
            ]
            ax_dag.legend(handles=legend_elems, loc="lower right")
            ax_dag.axis("off")
            
            st.pyplot(fig_dag)

    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi xử lý dữ liệu: {e}")

else:
    st.info("Vui lòng tải lên bộ dữ liệu (file .dta hoặc .csv) ở thanh bên trái để bắt đầu phân tích.")