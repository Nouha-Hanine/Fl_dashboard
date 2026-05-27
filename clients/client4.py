import flwr as fl
import pickle
import numpy as np
import pandas as pd
import grpc

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    fbeta_score,
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
    confusion_matrix
)

# =========================
# CONFIG
# =========================
CLIENT_ID = 4
THRESHOLD = 0.5


# =========================
# LOAD DATA FROM ORCHESTRATOR
# =========================
def get_data(client_id):

    path = f"orchestrator/client_data/client{client_id}.csv"

    print(f"[CLIENT {client_id}] Loading data from {path}")

    df = pd.read_csv(path, sep=";")  # ✅ IMPORTANT FIX

    target = "Progression_Status"

    X = pd.get_dummies(df.drop(columns=[target]))
    y = df[target]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


# =========================
# CLIENT FL
# =========================
class FlowerClient(fl.client.NumPyClient):

    def __init__(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.tau = 0.9
        self.alpha = 0.7
        self.threshold = THRESHOLD

    # ========================= FIT =========================
    def fit(self, parameters, config):

        print("\n--- [CLIENT 1] FIT ---")

        data = pickle.loads(parameters[0])

        gm_model = data["model"]
        K = data["K"]
        hyperparams = data["params"]

        # align features
        X_aligned = self.X_train.reindex(
            columns=gm_model.feature_names_in_,
            fill_value=0
        )

        # prediction global model
        proba = gm_model.predict_proba(X_aligned)[:, 1]
        y_pred = (proba > self.threshold).astype(int)

        f1 = fbeta_score(self.y_train, y_pred, beta=1)

        print(f"[CLIENT 1] F1 global model = {f1:.4f}")

        # adaptive threshold
        self.tau = self.alpha * self.tau + (1 - self.alpha) * f1

        bij = bool(f1 >= (self.tau - 0.03))

        tij_bytes = pickle.dumps([])

        # correction trees if bad performance
        if not bij:

            print("[CLIENT 1] Training correction trees...")

            weights = np.where(
                gm_model.predict(X_aligned) != self.y_train,
                2.0,
                1.0
            )

            new_rf = RandomForestClassifier(**hyperparams)
            new_rf.fit(X_aligned, self.y_train, sample_weight=weights)

            trees = sorted(
                new_rf.estimators_,
                key=lambda t: fbeta_score(
                    self.y_train,
                    (t.predict(X_aligned) > 0.5).astype(int),
                    beta=1
                ),
                reverse=True
            )[:K]

            tij_bytes = pickle.dumps(trees)

            print(f"[CLIENT 1] Sending {len(trees)} trees")

        else:
            print("[CLIENT 1] Model good, no update")

        return [], len(self.X_train), {
            "bij": bij,
            "tij": tij_bytes
        }

    # ========================= EVALUATE =========================
    def evaluate(self, parameters, config):

        print("\n--- [CLIENT 1] EVALUATE ---")

        data = pickle.loads(parameters[0])
        model = data["model"]

        X_test = self.X_test.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba > self.threshold).astype(int)

        acc = accuracy_score(self.y_test, y_pred)
        f1 = fbeta_score(self.y_test, y_pred, beta=1)
        auc = roc_auc_score(self.y_test, y_proba)

        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        spec = tn / (tn + fp + 1e-8)

        print("\n===== CLIENT 1 METRICS =====")
        print(f"Accuracy  : {acc:.4f}")
        print(f"F1-score  : {f1:.4f}")
        print(f"AUC       : {auc:.4f}")
        print(f"Specificity: {spec:.4f}")

        return float(1 - f1), len(self.X_test), {}


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    X_train, X_test, y_train, y_test = get_data(CLIENT_ID)

    print("[CLIENT 1] Connecting to server...")

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(
            X_train, X_test, y_train, y_test
        ).to_client()
    )