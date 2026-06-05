import streamlit as st
import pandas as pd
import requests
import os
import json
import time

st.set_page_config(
    page_title="FL Dashboard ",
    layout="wide",
    initial_sidebar_state="expanded"
)


# AUTO-REFRESH CONFIGURATION (TEMPS RÉEL)
if "training_active" not in st.session_state:
    st.session_state.training_active = False

st.markdown("""
<style>

:root {
    --text-color: #334155 !important; 
    --primary-color: #f97316 !important; 
    --background-color: #fdfbfc !important; 
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #fdfbfc 0%, #fbf7f4 50%, #f7f0ea 100%);
    color: #334155 !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}


.main-title {
    font-size: 3.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #eab308 0%, #f97316 50%, #dc2626 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    letter-spacing: -1px;
}

.sub-title {
    color: #64748b;
    font-size: 1.05rem;
    margin-bottom: 2rem;
    font-weight: 500;
}


.custom-card {
    background: rgba(253, 251, 252, 0.85);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(249, 115, 22, 0.12);
    border-radius: 24px;
    padding: 25px;
    box-shadow: 0 15px 35px rgba(249, 115, 22, 0.03), 0 1px 2px rgba(0,0,0,0.01);
}


.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 14px;
    border: none;
    font-weight: 700;
    font-size: 15px;
    letter-spacing: 0.3px;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}


div[data-testid="stHorizontalBlock"] div:nth-child(1) .stButton > button {
    background: linear-gradient(90deg, #eab308 0%, #f97316 100%);
    color: #ffffff !important;
    box-shadow: 0 8px 20px rgba(249, 115, 22, 0.2);
}

div[data-testid="stHorizontalBlock"] div:nth-child(1) .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 25px rgba(249, 115, 22, 0.3);
    filter: brightness(1.05);
}


div[data-testid="stHorizontalBlock"] div:nth-child(2) .stButton > button {
    background: #fff5f5;
    color: #dc2626 !important;
    border: 1.5px solid #fecaca;
}

div[data-testid="stHorizontalBlock"] div:nth-child(2) .stButton > button:hover {
    transform: translateY(-2px);
    background: #fef2f2;
    border-color: #fca5a5;
    box-shadow: 0 8px 15px rgba(220, 38, 38, 0.08);
}


[data-testid="metric-container"] {
    background: rgba(253, 251, 252, 0.9);
    border: 1px solid rgba(249, 115, 22, 0.1);
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.01);
}


.stTextArea textarea {
    border-radius: 14px !important;
    border: 1px solid #e2e8f0 !important;
    background-color: #1e293b !important; 
    color: #f8fafc !important;
    font-family: 'Fira Code', monospace;
    font-size: 13px;
    padding: 15px;
}

[data-testid="stTextArea"] label p {
    color: #334155 !important;
    font-weight: 600 !important;
}


/*BLOC SIDEBAR*/ 
  

section[data-testid="stSidebar"] {
    background: #ffedd5 !important;
    border-right: 1px solid #edf2f7;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2 {
    color: #0f172a !important;
    font-size: 1.4rem !important;
    font-weight: 800 !important;
}

section[data-testid="stSidebar"] label p {
    color: #475569 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}


.stSelectbox > div > div {
    background-color: #ffffff !important;
    color: #334155 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px;
}
.stSelectbox svg {
    fill: #f97316 !important;
}              


.streamlit-expanderHeader {
    font-size: 15px;
    font-weight: 600;
    color: #334155;
    background-color: #fbf7f4;
    border: 1px solid #eef2f6;
    border-radius: 12px;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #eef2f6;
}

.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
    color: #0f172a;
    letter-spacing: -0.3px;
}

hr {
    border-color: #eef2f6;
}


.glow {
    box-shadow: 0 0 30px rgba(249, 115, 22, 0.08);
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="custom-card glow">
        <div class="main-title">
             Federated Learning Dashboard 
        </div>

        <div class="sub-title">
            A real-time dashboard for orchestrating, monitoring,
            and visualizing Federated Learning training across distributed clients.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("# ⚙️ Configuration")

num_clients = st.sidebar.selectbox(
    "👥 Number of Clients",
    [2, 3, 5]
)

k_value = st.sidebar.selectbox(
    "🌳 K Trees",
    [3, 10, 20]
)

split_choice = st.sidebar.selectbox(
    "✂️ Split Dataset",
    [
        "500 / 235",
        "367 / 368"
    ]
)

if split_choice == "500 / 235":
    dataset_choice = st.sidebar.selectbox(
        "📂 Dataset For Clients",
        [
            "500_Data.csv",
            "235_Data.csv"
        ]
    )
else:
    dataset_choice = st.sidebar.selectbox(
        "📂 Dataset For Clients",
        [
            "367_Data.csv",
            "368_Data.csv"
        ]
    )

model_choice = st.sidebar.selectbox(
    "🤖 Pretrained Model",
    [
        "random2_forest_model_tcga+emtab500.pkl",
        "random2_forest_model_tcga+emtab367.pkl",
        "random2_forest_model_tcga+emtab235.pkl"
    ]
)

st.sidebar.markdown("---")

if st.session_state.training_active:
    st.sidebar.warning("🔄 Training in progress...")
else:
    st.sidebar.success("System Ready")

# =========================================================
# DATASET PREVIEW
# =========================================================
st.markdown(
    '<div class="section-title">📊 Dataset Preview</div>',
    unsafe_allow_html=True
)

dataset_path = "data/735_Data.csv"

if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
    c1, c2, c3 = st.columns(3)

    
    def custom_white_metric(col, label, value):
        col.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #dc2626 0%, #f97316 50%, #facc15 100%);
                border: 1px solid rgba(255, 255, 255, 0.2);
                padding: 24px;
                border-radius: 20px;
                text-align: center;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                box-shadow: 0 10px 25px rgba(249, 115, 22, 0.35);
                height: 140px;
            ">
                <div style="
                    color: rgba(255, 255, 255, 0.85); 
                    font-size: 13px; 
                    font-weight: 700; 
                    text-transform: uppercase; 
                    letter-spacing: 1px; 
                    margin-bottom: 8px;
                ">
                    {label}
                </div>
                <div style="
                    color: #ffffff; 
                    font-size: 42px; 
                    font-weight: 900;
                    line-height: 1;
                ">
                    {value}
                </div>
            </div>
        """, unsafe_allow_html=True)

    custom_white_metric(c1, "Rows", len(df))
    custom_white_metric(c2, "Columns", len(df.columns))
    custom_white_metric(c3, "Clients", num_clients)

    st.write("") 
    st.dataframe(df.head(), use_container_width=True)
else:
    st.error("Dataset not found")

st.write("")

# =========================================================
# CONTROL PANEL
# =========================================================
st.markdown('<div class="section-title">🎛️ Control Center</div>', unsafe_allow_html=True)
control4, control5 = st.columns(2)

with control4:
    if st.button("🚀 START FL TRAINING"):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/start_fl",
                json={
                    "num_clients": num_clients,
                    "k_value": k_value,
                    "model_choice": model_choice,
                    "dataset": dataset_choice,
                    "split_strategy": split_choice,
                }
            )
            if response.status_code == 200:
                st.session_state.training_active = True
                st.success("Federated Learning started")
                st.rerun()
            else:
                st.error(response.text)
        except Exception as e:
            st.error(f"Cannot start FL: {e}")

with control5:
    if st.button("🛑 STOP TRAINING"):
        try:
            response = requests.post("http://127.0.0.1:8000/stop_fl")
            st.session_state.training_active = False
            st.warning("Training stopped")
            st.rerun()
        except Exception as e:
            st.error(f"Stop failed: {e}")

# =========================================================
# GLOBAL METRICS
# =========================================================
st.markdown(
    '<div class="section-title">📈 Global Model Metrics</div>',
    unsafe_allow_html=True
)

metrics_path = "dashboard/global_metrics.json"

if os.path.exists(metrics_path):
    try:
        with open(metrics_path, "r") as f:
            content = f.read().strip()

        if content:
            data = json.loads(content)
            m1, m2, m3, m4, m5 = st.columns(5)

            def custom_white_metric(col, label, value):
                col.markdown(f"""
                    <div style="
    background: white;
    border: 2px solid black;
    padding: 18px;
    border-radius: 16px;
    text-align: left;
    box-shadow: 0 10px 25px rgba(179, 82, 19, 0.45);
">
    <div style="text-align: center;">
    <div style="color: #475569; font-size: 15px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">{label}</div>
    <div style="background: linear-gradient(135deg, #f97316 0%, #dc2626 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 30px; font-weight: 800; display: inline-block;">{value}</div>
</div></div>
                """, unsafe_allow_html=True)

            custom_white_metric(m1, "Accuracy", round(data.get("accuracy", 0), 3))
            custom_white_metric(m2, "F1-score", round(data.get("f1", 0), 3))
            custom_white_metric(m3, "Precision", round(data.get("precision", 0), 3))
            custom_white_metric(m4, "Sensitivity", round(data.get("sensitivity", 0), 3))
            custom_white_metric(m5, "AUC", round(data.get("auc", 0), 3))
        else:
            st.info("Waiting for metrics...")
    except Exception as e:
        st.error(f"Metrics Error: {e}")
else:
    st.info("No metrics generated yet")

st.write("")
st.markdown("---")

# =========================================================
# SERVER LOGS
# =========================================================
st.markdown('<div class="section-title">🖥️ Server Logs</div>', unsafe_allow_html=True)
log_path = "logs/server.log"

if os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        logs = f.read()

    st.text_area(
        label="Server logs",
        value=logs,
        height=300
    )
    
    if "TRAINING FINISHED" in logs or "ERROR:" in logs:
        st.session_state.training_active = False
else:
    st.info("No logs yet")

# =========================================================
# CLIENT RESULTS
# =========================================================
st.markdown(
    '<div class="section-title"> 📊 Client Visualizations</div>',
    unsafe_allow_html=True
)

from PIL import Image

for i in range(1, num_clients + 1):
    with st.expander(f"📊 Client {i} Results"):
        result_folder = f"results/client{i}"

        if os.path.exists(result_folder):
            files = os.listdir(result_folder)
            images = [f for f in files if f.endswith(".png")]

            if len(images) == 0:
                st.info("No graphs generated yet")
            else:
                cols = st.columns(2)
                for idx, img in enumerate(images):
                    img_path = f"{result_folder}/{img}"
                    try:
                        with Image.open(img_path) as v_img:
                            v_img.verify() 
                        
                        cols[idx % 2].image(
                            img_path,
                            caption=img,
                            width="stretch"
                        )
                    except Exception:
                        pass
        else:
            st.warning("No results folder found")

st.write("")
st.markdown("---")

# =========================================================
# ENGINE REFRESH
# =========================================================
if st.session_state.training_active:
    time.sleep(2)
    st.rerun()

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <center>
        <span style='color:#94a3b8; font-size: 13px;'>
             FL Dashboard • Real-Time Federated Learning Monitoring System
        </span>
    </center>
    """,
    unsafe_allow_html=True
)