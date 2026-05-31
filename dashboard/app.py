
import streamlit as st
import pandas as pd
import requests
import os
import json
import time
# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="FL Dashboard ",
    layout="wide",
    initial_sidebar_state="expanded"
)
# =========================================================
# AUTO-REFRESH CONFIGURATION (TEMPS RÉEL)
# =========================================================
# Si l'entraînement est actif, on force la page à se recharger toutes les 2 secondes
if "training_active" not in st.session_state:
    st.session_state.training_active = False

# =========================================================
# CUSTOM CSS (ULTRA MODERN UI)
# =========================================================
st.markdown("""
<style>

/* Force les variables globales de Streamlit à utiliser du blanc par défaut */
:root {
    --text-color: #ffffff !important;
    --primary-color: #2563eb !important;
}

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
    color: #ffffff !important;
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

/* =========================================================
   BLOC METRICS CORRIGÉ (Rows, Columns, Accuracy, F1...) 
   ========================================================= */

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
}

/* Petits titres en haut des metrics (ex: Rows, Columns, Accuracy) -> Gris clair argenté */
[data-testid="metric-container"] label, 
[data-testid="metric-container"] [data-testid="stMetricLabel"],
[data-testid="metric-container"] [data-testid="stMetricLabel"] > div,
[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
}

/* Grands chiffres et valeurs (ex: 735, 10, 0.685) -> Blanc pur */
[data-testid="metric-container"] [data-testid="stMetricValue"], 
[data-testid="metric-container"] [data-testid="stMetricValue"] > div,
[data-testid="metric-container"] [data-testid="stMetricValue"] div {
    color: #ffffff !important;
    font-weight: 700 !important;
    -webkit-text-fill-color: #ffffff !important; /* Force le rendu webkit */
}

/* Titre "Server logs" -> Blanc pur */
[data-testid="stTextArea"] label, 
[data-testid="stTextArea"] label p,
.stTextArea label p {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* =========================================================
   BLOC SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background: #0b1120 !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Titre "⚙️ Configuration" -> Blanc pur */
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h2 div,
section[data-testid="stSidebar"] h1 {
    color: #ffffff !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* Textes au-dessus des menus déroulants -> Gris clair */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] label p {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* Flèches et icône de réduction de la Sidebar -> Blanc pur */
section[data-testid="stSidebar"] svg,
[data-testid="stSidebarCollapseButton"] svg {
    fill: #ffffff !important;
    color: #ffffff !important;
}

/* Selectbox (Champs de sélection) */
.stSelectbox > div > div {
    background-color: #111827;
    color: white;
    border-radius: 12px;
}
.stSelectbox svg {
    fill: #ffffff !important;
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
             Federated Learning Dashboard 
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
    [2, 3, 4, 5]
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
        "random_forest_model_tcga+emtab235.pkl",
        "random_forest_model_tcga+emtab367.pkl",
        "random_forest_model_tcga+emtab500.pkl",
        "random2_forest_model_tcga+emtab500.pkl",
        "random2_forest_model_tcga+emtab367.pkl",
        "random2_forest_model_tcga+emtab235.pkl"
    ]
)

st.sidebar.markdown("---")
st.sidebar.success("System Ready")

# Changement d'état visuel de la sidebar en direct
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

    # --- FONCTION LOCALE POUR GENERER LE METRIC EN BLANC PUR ---
    def custom_white_metric(col, label, value):
        col.markdown(f"""
            <div style="
                background: rgba(255,255,255,0.05);
                border: 1px solid rgba(255,255,255,0.08);
                padding: 15px;
                border-radius: 14px;
                text-align: left;
            ">
                <div style="color: #cbd5e1; font-size: 14px; font-weight: 500; margin-bottom: 5px;">{label}</div>
                <div style="color: #ffffff; font-size: 28px; font-weight: 700;">{value}</div>
            </div>
        """, unsafe_allow_html=True)

    # Application de la fonction sur tes 3 colonnes de Preview
    custom_white_metric(c1, "Rows", len(df))
    custom_white_metric(c2, "Columns", len(df.columns))
    custom_white_metric(c3, "Clients", num_clients)

    st.write("") # Petit espace visuel entre les metrics et le tableau
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
                # Ajout de l'activation du temps réel
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
            # Désactivation immédiate au clic
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

            # --- FONCTION LOCALE POUR GENERER LE METRIC EN BLANC PUR ---
            def custom_white_metric(col, label, value):
                col.markdown(f"""
                    <div style="
                        background: rgba(255,255,255,0.05);
                        border: 1px solid rgba(255,255,255,0.08);
                        padding: 15px;
                        border-radius: 14px;
                        text-align: left;
                    ">
                        <div style="color: #cbd5e1; font-size: 14px; font-weight: 500; margin-bottom: 5px;">{label}</div>
                        <div style="color: #ffffff; font-size: 28px; font-weight: 700;">{value}</div>
                    </div>
                """, unsafe_allow_html=True)

            # Affichage forcé en blanc pur dans chaque colonne
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
    
    # Sécurité : Coupe l'auto-refresh automatiquement si ces mots-clés apparaissent
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

from PIL import Image  # Ajout requis pour la vérification de sécurité

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
                        # SÉCURITÉ : On vérifie si l'image est valide et totalement écrite sur le disque
                        with Image.open(img_path) as v_img:
                            v_img.verify() 
                        
                        # Si elle est valide, on l'affiche avec la nouvelle syntaxe demandée par Streamlit
                        cols[idx % 2].image(
                            img_path,
                            caption=img,
                            width="stretch"  # Remplace proprement 'use_container_width=True'
                        )
                    except Exception:
                        # Si l'image est corrompue ou en cours d'écriture, on ignore temporairement sans planter
                        pass

        else:
            st.warning("No results folder found")


st.write("")
st.markdown("---")

# =========================================================
# 5. LE MOTEUR DU TEMPS REEL (TOUT EN BAS DU CODE)
# =========================================================
if st.session_state.training_active:
    time.sleep(2)  # Attendre 2 secondes avant de rafraîchir
    st.rerun()     # Réexécute le script pour mettre à jour les logs et graphes
# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <center>
        <span style='color:#94a3b8'>
            🚀 FL Dashboard • Real-Time Federated Learning Monitoring System
        </span>
    </center>
    """,
    unsafe_allow_html=True
)