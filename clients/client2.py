import flwr as fl
import pickle
import numpy as np
import pandas as pd
import grpc
import json
import requests

from sklearn.metrics import (
    fbeta_score, accuracy_score, recall_score,
    precision_score, roc_auc_score, confusion_matrix, roc_curve
)

from sklearn.ensemble import RandomForestClassifier


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, client_id):

        self.client_id = client_id
        self.X_train_local = None
        self.X_test_local = None
        self.y_train = None
        self.y_test = None

        self.X_external = None
        self.y_external = None

        self.tau = 0.9
        self.alpha = 0.7
        self.threshold = 0.5

    # ========================= FIT =========================
    def fit(self, parameters, config):

        print("\n--- FIT PHASE ---")

        data = pickle.loads(parameters[0])

        gm_model = data["model"]
        K = data["K"]
        hyperparams = data["params"]

        dataset = data.get("data", None)

        # -------- receive dataset from dashboard --------
        if dataset:

            self.X_train_local = dataset["X_train"]
            self.X_test_local = dataset["X_test"]
            self.y_train = dataset["y_train"]
            self.y_test = dataset["y_test"]

            self.X_external = dataset.get("X_external", None)
            self.y_external = dataset.get("y_external", None)

            print("[CLIENT] Dataset received from dashboard.")

        # -------- align features --------
        X_aligned = self.X_train_local.reindex(
            columns=gm_model.feature_names_in_,
            fill_value=0
        )

        proba = gm_model.predict_proba(X_aligned)[:, 1]
        y_pred = (proba > self.threshold).astype(int)

        f1 = fbeta_score(self.y_train, y_pred, beta=1)

        self.tau = self.alpha * self.tau + (1 - self.alpha) * f1

        bij = bool(f1 >= (self.tau - 0.03))

        tij_bytes = pickle.dumps([])

        # -------- correction trees --------
        if not bij:

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

        return [], len(self.X_train_local), {
            "bij": bij,
            "tij": tij_bytes
        }

    # ========================= EVALUATE =========================
    def evaluate(self, parameters, config):

        print("\n--- EVALUATION PHASE ---")

        data = pickle.loads(parameters[0])
        model = data["model"]

        # ================= LOCAL TEST =================
        X_test = self.X_test_local.reindex(
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

        # ================= CROSS CLIENT =================
        cross_metrics = {}

        if self.X_external is not None:

            X_ext = self.X_external.reindex(
                columns=model.feature_names_in_,
                fill_value=0
            )

            y_ext_proba = model.predict_proba(X_ext)[:, 1]
            y_ext_pred = (y_ext_proba > self.threshold).astype(int)

            cross_metrics = {
                "acc": accuracy_score(self.y_external, y_ext_pred),
                "f1": fbeta_score(self.y_external, y_ext_pred, beta=1),
                "precision": precision_score(self.y_external, y_ext_pred),
                "recall": recall_score(self.y_external, y_ext_pred),
                "auc": roc_auc_score(self.y_external, y_ext_proba)
            }

        # ================= GRAPHS DATA =================
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)

        confusion = [[tn, fp], [fn, tp]]

        payload_to_dashboard = {
            "client_id": self.client_id,
            "local_metrics": {
                "acc": acc,
                "f1": f1,
                "auc": auc,
                "specificity": spec
            },
            "cross_metrics": cross_metrics,
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist()
            },
            "confusion_matrix": confusion
        }

        # ================= SEND TO DASHBOARD =================
        try:
            requests.post(
                "http://127.0.0.1:5000/update-dashboard",
                data=pickle.dumps(payload_to_dashboard),
                headers={"Content-Type": "application/octet-stream"}
            )

            print("[CLIENT] Metrics sent to dashboard.")

        except Exception as e:
            print("Dashboard send error:", e)

        return float(1 - f1), len(self.X_test_local), {}
