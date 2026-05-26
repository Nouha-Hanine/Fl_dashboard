import streamlit as st
import requests
import time
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="FL Dashboard", layout="wide")

st.title("🔥 Federated Learning Dashboard")

API_URL = "http://127.0.0.1:8000"

# =========================
# SIDEBAR CONTROL PANEL
# =========================
st.sidebar.header("⚙️ Control Panel")

clients = st.sidebar.selectbox("Number of Clients", [2, 3, 4, 5])

if st.sidebar.button("🚀 Start FL"):
    r = requests.post(f"{API_URL}/start_fl")
    st.sidebar.success(r.json()["status"])

if st.sidebar.button("🛑 Stop FL"):
    r = requests.post(f"{API_URL}/stop_fl")
    st.sidebar.warning(r.json()["status"])

# =========================
# MAIN DASHBOARD
# =========================
st.subheader("📊 Training Monitor")

placeholder = st.empty()

# Fake data (remplacé plus tard par ton serveur)
rounds = list(range(1, 11))
accuracy = [0.6, 0.65, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.83]
f1 = [0.55, 0.6, 0.63, 0.68, 0.7, 0.72, 0.75, 0.77, 0.79, 0.81]

df = pd.DataFrame({
    "Round": rounds,
    "Accuracy": accuracy,
    "F1-score": f1
})

placeholder.dataframe(df)

# =========================
# GRAPHS
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Accuracy")
    fig, ax = plt.subplots()
    ax.plot(rounds, accuracy)
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    st.pyplot(fig)

with col2:
    st.subheader("📈 F1 Score")
    fig, ax = plt.subplots()
    ax.plot(rounds, f1)
    ax.set_xlabel("Round")
    ax.set_ylabel("F1")
    st.pyplot(fig)

# =========================
# STATUS PANEL
# =========================
st.subheader("📡 System Status")

st.info("Dashboard connected to FL system")
st.write("Clients selected:", clients)