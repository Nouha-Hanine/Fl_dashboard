import streamlit as st
import pandas as pd
import requests
import os
import json

# =========================
# CONFIG UI
# =========================
st.set_page_config(
    page_title="FL Dashboard PRO",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# STYLE (MODERNE)
# =========================
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: #ffffff;
}
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
    background-color: #4F8BF9;
    color: white;
}
.block-container {
    padding-top: 2rem;
}
.metric-box {
    background-color: #161b22;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("🚀 Federated Learning Control Dashboard")

st.write("Control dataset, clients, models, training & results in one place.")

# =========================
# SIDEBAR CONFIG
# =========================
st.sidebar.header("⚙️ Configuration Panel")

num_clients = st.sidebar.selectbox("👥 Number of Clients", [2, 3, 4, 5])
k_value = st.sidebar.selectbox("🌳 K (trees per round)", [3, 10, 20])

model_choice = st.sidebar.selectbox(
    "🤖 Pretrained Model",
    [
        "random_forest_model1.pkl",
        "random_forest_model2.pkl",
        "random_forest_model3.pkl"
    ]
)

split_choice = st.sidebar.selectbox(
    "✂️ Dataset Split Strategy",
    [
        "367 / 368",
        "500 / 235",
        "Auto (equal split)"
    ]
)

script_choice = st.sidebar.selectbox(
    "📜 Client Script Strategy",
    [
        "client.py (standard)",
        "client_weighted.py",
        "client_advanced.py"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info("All configuration will be sent to orchestrator API")

# =========================
# DATASET PREVIEW
# =========================
st.subheader("📊 Dataset Preview")

dataset_path = "data/735_Data.csv"

if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
    st.success(f"Dataset loaded: {len(df)} rows")

    st.dataframe(df.head(), use_container_width=True)
else:
    st.error("Dataset not found")

st.markdown("---")

# =========================
# CONTROL PANEL (BUTTONS)
# =========================
st.subheader("🎛️ Control Panel")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("✂️ SPLIT DATASET"):
        response = requests.post(
            "http://127.0.0.1:8000/split",
            json={
                "num_clients": num_clients,
                "split_strategy": split_choice
            }
        )
        st.success(response.json())

with col2:
    if st.button("📤 SEND MODEL TO SERVER"):
        response = requests.post(
            "http://127.0.0.1:8000/load_model",
            json={"model": model_choice}
        )
        st.success(response.json())

with col3:
    if st.button("🚀 START FL TRAINING"):
        response = requests.post("http://127.0.0.1:8000/start_fl")
        st.success(response.json())

col4, col5 = st.columns(2)

with col4:
    if st.button("📦 SEND ALL CLIENT DATA"):
        response = requests.post(
            "http://127.0.0.1:8000/distribute_data"
        )
        st.success(response.json())

with col5:
    if st.button("🛑 STOP FL"):
        response = requests.post("http://127.0.0.1:8000/stop_fl")
        st.warning(response.json())

st.markdown("---")

# =========================
# GLOBAL METRICS
# =========================
st.subheader("📈 Global Model Metrics")

metrics_path = "dashboard/global_metrics.json"

if os.path.exists(metrics_path):
    try:
        with open(metrics_path, "r") as f:
            content = f.read().strip()

        if content:
            data = json.loads(content)

            c1, c2, c3, c4, c5 = st.columns(5)

            c1.metric("Accuracy", round(data.get("accuracy", 0), 3))
            c2.metric("F1-score", round(data.get("f1", 0), 3))
            c3.metric("Precision", round(data.get("precision", 0), 3))
            c4.metric("Sensitivity", round(data.get("sensitivity", 0), 3))
            c5.metric("AUC", round(data.get("auc", 0), 3))
        else:
            st.info("Waiting for training results...")

    except Exception as e:
        st.error(f"Error loading metrics: {e}")
else:
    st.info("No training results yet")

st.markdown("---")

# =========================
# CLIENT RESULTS VISUALIZATION
# =========================
st.subheader("🧠 Client Results Visualization")

for i in range(1, num_clients + 1):

    with st.expander(f"📊 Client {i} Results", expanded=False):

        folder = f"results/client{i}"

        if os.path.exists(folder):

            files = os.listdir(folder)
            pngs = [f for f in files if f.endswith(".png")]

            if len(pngs) == 0:
                st.info("No graphs available yet")
            else:
                cols = st.columns(2)

                for idx, img in enumerate(pngs):
                    cols[idx % 2].image(
                        f"{folder}/{img}",
                        caption=img,
                        use_container_width=True
                    )
        else:
            st.warning("Client results not found")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("🚀 FL Dashboard PRO | Designed for Federated Learning Control & Visualization")