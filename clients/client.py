import flwr as fl
import pickle
import numpy as np
import pandas as pd
import sys
import os
import grpc

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    fbeta_score,
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
    confusion_matrix
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# =====================================================
# CLIENT ID FROM DASHBOARD
# =====================================================
CLIENT_ID = sys.argv[1]

print(f"\n[CLIENT {CLIENT_ID}] Starting...")


# =====================================================
# LOAD LOCAL CLIENT DATA
# =====================================================
def get_data():

    try:

        dataset_path = (
            f"orchestrator/client_data/client{CLIENT_ID}.csv"
        )

        print(f"[CLIENT {CLIENT_ID}] Loading: {dataset_path}")

        # 🔥 FIX ONLY HERE
        df = pd.read_csv(dataset_path, sep=";")

        target = "Progression_Status"

        X_raw = pd.get_dummies(
            df.drop(columns=[target])
        )

        y = df[target]

        return train_test_split(
            X_raw,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

    except Exception as e:

        print(f"[CLIENT {CLIENT_ID}] Error loading data: {e}")

        return None, None, None, None


# =====================================================
# FLOWER CLIENT
# =====================================================
class FlowerClient(fl.client.NumPyClient):

    def __init__(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        self.X_train_local = X_train
        self.X_test_local = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.tau = 0.9
        self.alpha = 0.7
        self.threshold = 0.5

    # =================================================
    # FIT
    # =================================================
    def fit(self, parameters, config):

        print(f"\n--- [CLIENT {CLIENT_ID}] FIT PHASE ---")

        data = pickle.loads(parameters[0])

        gm_model = data["model"]

        K = data["K"]

        hyperparams = data["params"]

        if "class_weight" not in hyperparams:
            hyperparams["class_weight"] = "balanced"

        print(
            f"[CLIENT {CLIENT_ID}] "
            f"Received global model "
            f"({len(gm_model.estimators_)} trees)"
        )

        # =============================================
        # FEATURE ALIGNMENT
        # =============================================
        X_aligned = self.X_train_local.reindex(
            columns=gm_model.feature_names_in_,
            fill_value=0
        )

        # =============================================
        # PREDICTION
        # =============================================
        proba = gm_model.predict_proba(
            X_aligned
        )[:, 1]

        y_pred = (
            proba > self.threshold
        ).astype(int)

        performance = fbeta_score(
            self.y_train,
            y_pred,
            beta=1
        )

        print(
            f"[CLIENT {CLIENT_ID}] "
            f"Local F1 = {performance:.4f}"
        )

        # =============================================
        # DYNAMIC THRESHOLD
        # =============================================
        self.tau = (
            self.alpha * self.tau
            + (1 - self.alpha) * performance
        )

        TOLERANCE = 0.03

        bij = bool(
            performance >= (self.tau - TOLERANCE)
        )

        tij_bytes = pickle.dumps([])

        # =============================================
        # GENERATE CORRECTIONS
        # =============================================
        if not bij:

            print(
                f"[CLIENT {CLIENT_ID}] "
                f"Generating corrective trees..."
            )

            weights = np.where(
                gm_model.predict(X_aligned)
                != self.y_train,
                2.0,
                1.0
            )

            new_rf = RandomForestClassifier(
                **hyperparams
            )

            new_rf.fit(
                X_aligned,
                self.y_train,
                sample_weight=weights
            )

            trees = sorted(
                new_rf.estimators_,
                key=lambda t: fbeta_score(
                    self.y_train,
                    (
                        t.predict(X_aligned) > 0.5
                    ).astype(int),
                    beta=1
                ),
                reverse=True
            )[:K]

            tij_bytes = pickle.dumps(trees)

            print(
                f"[CLIENT {CLIENT_ID}] "
                f"{len(trees)} trees sent"
            )

        else:

            print(
                f"[CLIENT {CLIENT_ID}] "
                f"No corrections needed"
            )

        return [], len(self.X_train_local), {
            "bij": bij,
            "tij": tij_bytes
        }

    # =================================================
    # EVALUATE
    # =================================================
    def evaluate(self, parameters, config):

        print(f"\n--- [CLIENT {CLIENT_ID}] EVALUATION ---")

        data = pickle.loads(parameters[0])

        model = data["model"]

        X_test_aligned = self.X_test_local.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        y_proba = model.predict_proba(
            X_test_aligned
        )[:, 1]

        y_pred = (
            y_proba > self.threshold
        ).astype(int)

        # ✅ FIX APPLIED HERE
        acc = accuracy_score(
            self.y_test,
            y_pred
        )

        sens = recall_score(
            self.y_test,
            y_pred
        )

        prec = precision_score(
            self.y_test,
            y_pred
        )

        f1 = fbeta_score(
            self.y_test,
            y_pred,
            beta=1
        )

        auc = roc_auc_score(
            self.y_test,
            y_proba
        )

        tn, fp, fn, tp = confusion_matrix(
            self.y_test,
            y_pred
        ).ravel()

        spec = (
            tn / (tn + fp)
            if (tn + fp) > 0 else 0
        )

        custom_loss = 1.0 - f1

        print("\n========== RESULTS ==========")

        print(f"Loss         : {custom_loss:.4f}")
        print(f"F1 Score     : {f1:.4f}")
        print(f"Accuracy     : {acc:.4f}")
        print(f"Sensitivity  : {sens:.4f}")
        print(f"Precision    : {prec:.4f}")
        print(f"Specificity  : {spec:.4f}")
        print(f"ROC AUC      : {auc:.4f}")

        return float(custom_loss), len(self.X_test_local), {}


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":

    X_train, X_test, y_train, y_test = get_data()

    if X_train is not None:

        print(
            f"[CLIENT {CLIENT_ID}] "
            f"Connecting to FL server..."
        )

        try:

            fl.client.start_client(
                server_address="127.0.0.1:8080",
                client=FlowerClient(
                    X_train,
                    X_test,
                    y_train,
                    y_test
                ).to_client()
            )

        except grpc.RpcError:

            print(
                f"[CLIENT {CLIENT_ID}] "
                f"Server stopped"
            )

        except Exception as e:

            print(
                f"[CLIENT {CLIENT_ID}] Error: {e}"
            )