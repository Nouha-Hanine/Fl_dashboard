
import streamlit as st
import pandas as pd
import requests
import os
import json

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="FL Dashboard PRO",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS (ULTRA MODERN UI)
# =========================================================
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

/* Main Background */
.stApp {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #111827 30%,
        #1e293b 100%
    );
    color: white;
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* Titles */
.main-title {
    font-size: 3rem;
    font-weight: 800;
    color: white;
    margin-bottom: 0.2rem;
}

.sub-title {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* Cards */
.custom-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 25px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}

/* Buttons */
.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 16px;
    border: none;
    font-weight: 700;
    font-size: 15px;
    transition: 0.3s ease;
    background: linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    );
    color: white;
    box-shadow: 0 8px 20px rgba(59,130,246,0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    background: linear-gradient(
        90deg,
        #3b82f6,
        #8b5cf6
    );
}

/* Metrics */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0b1120;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Selectbox */
.stSelectbox > div > div {
    background-color: #111827;
    color: white;
    border-radius: 12px;
}

/* Expander */
.streamlit-expanderHeader {
    font-size: 18px;
    font-weight: bold;
    color: white;
}

/* Tables */
[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
}

/* Section titles */
.section-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 1rem;
    color: white;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.1);
}

/* Success box */
.success-box {
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.4);
    padding: 15px;
    border-radius: 15px;
    color: #6ee7b7;
}

/* Glow effect */
.glow {
    box-shadow: 0 0 25px rgba(124,58,237,0.45);
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
            🚀 Federated Learning Dashboard PRO
        </div>

        <div class="sub-title">
            Modern orchestration platform for Federated Learning,
            dataset management, client monitoring and real-time metrics visualization.
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
    [2,3,4,5]
)

k_value = st.sidebar.selectbox(
    "🌳 K Trees",
    [3,10,20]
)

split_choice = st.sidebar.selectbox(
    "✂️ Split Dataset",
    [
        "367 / 368",
        "500 / 235"
    ]
)

dataset_choice = st.sidebar.selectbox(
    "📂 Dataset For Clients",
    [
        "368_Data.csv",
        "500_Data.csv",
        "235_Data.csv"
    ]
)

script_choice = st.sidebar.selectbox(
    "📜 Client Strategy",
    [
        "split1.py",
        "split2.py",
        "split3.py"
    ]
)

model_choice = st.sidebar.selectbox(
    "🤖 Pretrained Model",
    [ "random_forest_model_tcga+emtab235.pkl",
      "random_forest_model_tcga+emtab367.pkl",
      "random_forest_model_tcga+emtab500.pkl" ]
)
client_split_choice = st.sidebar.selectbox(
    "🧩 Client Split Strategy",
    [
        "client.py"
    ]
)

st.sidebar.markdown("---")

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

    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Clients", num_clients)

    st.dataframe(
        df.head(),
        use_container_width=True,
        height=250
    )

else:
    st.error("Dataset not found")

st.write("")

# =========================================================
# CONTROL PANEL
# =========================================================
st.markdown(
    '<div class="section-title">🎛️ Control Center</div>',
    unsafe_allow_html=True
)


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
                    "split_strategy": split_choice,
                    "dataset": dataset_choice,
                    "client_split_strategy": client_split_choice,
                        
                }
            )

            st.success("Federated Learning started")

        except:
            st.error("Cannot start FL")

with control5:

    if st.button("🛑 STOP TRAINING"):

        try:
            response = requests.post(
                "http://127.0.0.1:8000/stop_fl"
            )

            st.warning("Training stopped")

        except:
            st.error("Stop failed")

st.write("")
st.markdown("---")

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

            m1,m2,m3,m4,m5 = st.columns(5)

            m1.metric(
                "Accuracy",
                round(data.get("accuracy",0),3)
            )

            m2.metric(
                "F1-score",
                round(data.get("f1",0),3)
            )

            m3.metric(
                "Precision",
                round(data.get("precision",0),3)
            )

            m4.metric(
                "Sensitivity",
                round(data.get("sensitivity",0),3)
            )

            m5.metric(
                "AUC",
                round(data.get("auc",0),3)
            )

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
st.markdown(
    '<div class="section-title">🖥️ Server Logs</div>',
    unsafe_allow_html=True
)

log_path = "logs/server.log"

if os.path.exists(log_path):

    with open(log_path, "r", encoding="utf-8") as f:

        logs = f.read()

    st.text_area(
        "",
        logs,
        height=300
    )

else:

    st.info("No logs yet")

st.write("")
st.markdown("---")

# =========================================================
# CLIENT RESULTS
# =========================================================
st.markdown(
    '<div class="section-title">🧠 Client Visualizations</div>',
    unsafe_allow_html=True
)

for i in range(1, num_clients + 1):

    with st.expander(
        f"📊 Client {i} Results",
        expanded=False
    ):

        result_folder = f"results/client{i}"

        if os.path.exists(result_folder):

            files = os.listdir(result_folder)

            images = [
                f for f in files
                if f.endswith(".png")
            ]

            if len(images) == 0:

                st.info("No graphs generated yet")

            else:

                cols = st.columns(2)

                for idx, img in enumerate(images):

                    cols[idx % 2].image(
                        f"{result_folder}/{img}",
                        caption=img,
                        use_container_width=True
                    )

        else:

            st.warning("No results folder found")

st.write("")
st.markdown("---")

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <center>
        <span style='color:#94a3b8'>
            🚀 FL Dashboard PRO • Real-Time Federated Learning Monitoring System
        </span>
    </center>
    """,
    unsafe_allow_html=True
)

