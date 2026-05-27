import flwr as fl
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, accuracy_score, roc_auc_score, confusion_matrix


# =========================
# CONFIG
# =========================
CLIENT_ID = int(os.getenv("CLIENT_ID", 1))
THRESHOLD = 0.5


# =========================
# LOAD DATA
# =========================
def get_data(client_id):

    path = f"orchestrator/client_data/client{client_id}.csv"

    print(f"[CLIENT {client_id}] Loading data from {path}")

    df = pd.read_csv(path)  # ✅ NO NEED sep=";" after fix

    df.columns = df.columns.str.strip()

    target = "Progression_Status"

    if target not in df.columns:
        raise ValueError(f"Missing target column: {target}")

    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X)

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


# =========================
# CLIENT
# =========================
class FlowerClient(fl.client.NumPyClient):

    def __init__(self, X_train, X_test, y_train, y_test, client_id):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.client_id = client_id

        self.threshold = THRESHOLD

    # ========================= FIT =========================
    def fit(self, parameters, config):

        print(f"\n--- CLIENT {self.client_id} FIT ---")

        # Flower RandomForest: model envoyé directement
        model = parameters[0]

        X_aligned = self.X_train.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        proba = model.predict_proba(X_aligned)[:, 1]
        y_pred = (proba > self.threshold).astype(int)

        f1 = fbeta_score(self.y_train, y_pred, beta=1)

        print(f"[CLIENT {self.client_id}] F1 = {f1:.4f}")

        # pas de modification complexe → simple FL baseline
        return [], len(self.X_train), {}

    # ========================= EVALUATE =========================
    def evaluate(self, parameters, config):

        print(f"\n--- CLIENT {self.client_id} EVALUATE ---")

        model = parameters[0]

        X_test = self.X_test.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba > self.threshold).astype(int)

        acc = accuracy_score(self.y_test, y_pred)
        f1 = fbeta_score(self.y_test, y_pred, beta=1)
        auc = roc_auc_score(self.y_test, proba)

        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        spec = tn / (tn + fp + 1e-8)

        print(f"[CLIENT {self.client_id}] ACC={acc:.4f} F1={f1:.4f} AUC={auc:.4f}")

        return float(1 - f1), len(self.X_test), {}


# ========================= MAIN
# =========================
if __name__ == "__main__":

    X_train, X_test, y_train, y_test = get_data(CLIENT_ID)

    print(f"[CLIENT {CLIENT_ID}] Connecting to server...")

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(
            X_train, X_test, y_train, y_test, CLIENT_ID
        ).to_client()
    )