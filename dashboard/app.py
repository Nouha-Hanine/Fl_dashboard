import streamlit as st
import pandas as pd
import requests
import os
import json

st.set_page_config(
    page_title="FL Dashboard",
    layout="wide"
)

st.title("Federated Learning Dashboard")

st.sidebar.header("Configuration")

# choix nombre clients
num_clients = st.sidebar.selectbox(
    "Number of Clients",
    [2,3,4,5]
)

# choix K
k_value = st.sidebar.selectbox(
    "K value",
    [3,10,20]
)

# choix modèle
model = st.sidebar.selectbox(
    "Pretrained Model",
    [
        "random_forest_model1.pkl",
        "random_forest_model2.pkl",
        "random_forest_model3.pkl"
    ]
)

st.sidebar.write("---")

# dataset
dataset_path="data/735_Data.csv"

if os.path.exists(dataset_path):

    df=pd.read_csv(dataset_path)

    st.success(
        f"Dataset loaded : {len(df)} patients"
    )

    st.write(df.head())

else:
    st.error("Dataset missing")

st.write("---")

col1,col2=st.columns(2)

with col1:

    if st.button("START FL"):

        response=requests.post(
            "http://127.0.0.1:8000/start_fl"
        )

        st.success(response.json())

with col2:

    if st.button("STOP FL"):

        response=requests.post(
            "http://127.0.0.1:8000/stop_fl"
        )

        st.warning(response.json())

st.write("---")

st.header("Global Metrics")

if os.path.exists("results/metrics.json"):

    with open("results/metrics.json") as f:

        data=json.load(f)

    c1,c2,c3,c4,c5=st.columns(5)

    c1.metric(
        "Accuracy",
        round(data["accuracy"],3)
    )

    c2.metric(
        "F1",
        round(data["f1"],3)
    )

    c3.metric(
        "Precision",
        round(data["precision"],3)
    )

    c4.metric(
        "Recall",
        round(data["recall"],3)
    )

    c5.metric(
        "AUC",
        round(data["auc"],3)
    )

else:

    st.info(
        "No metrics yet"
    )


st.write("---")

st.header("Client Results")

for i in range(1,num_clients+1):

    st.subheader(
        f"Client {i}"
    )

    result_folder=f"results/client{i}"

    if os.path.exists(result_folder):

        imgs=os.listdir(result_folder)

        for img in imgs:

            if img.endswith(".png"):

                st.image(
                    f"{result_folder}/{img}"
                )

    else:

        st.write(
            "No results yet"
        )