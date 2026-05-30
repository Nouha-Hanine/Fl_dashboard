import flwr as fl
import joblib
import pickle
import pandas as pd
import warnings
import os
import json
import logging

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# LOG SYSTEM (Correction : Utiliser uniquement les print redirigés par FastAPI)
# =========================================================
def log(msg):
    print(msg, flush=True) # flush=True force l'écriture immédiate dans le terminal/log

# =========================================================
# LOAD CONFIG (Correction : Gestion d'erreur si le fichier est absent au départ)
# =========================================================
config_path = os.path.join(os.path.dirname(__file__), "config.json")

if not os.path.exists(config_path):
    log("[SERVER ERROR] config.json introuvable. En attente du signal du dashboard...")
    import sys
    sys.exit(1)

with open(config_path, "r") as f:
    CONFIG = json.load(f)

MODEL_PATH = CONFIG["model"]
K_VALUE = CONFIG["k"]
NB_CLIENTS = CONFIG["clients"]

log("===================================")
log("FL CONFIG")
log(f"Model: {MODEL_PATH}")
log(f"K: {K_VALUE}")
log(f"Clients: {NB_CLIENTS}")
log("===================================")


# =========================================================
# LOAD MODEL
# =========================================================
try:
    GM = joblib.load(MODEL_PATH)
    log(f"[SERVER] Initial model loaded ({len(GM.estimators_)} trees)")
except Exception as e:
    raise FileNotFoundError(f"Model not found: {e}")


# =========================================================
# LOAD TEST DATA
# =========================================================
test_data = pd.read_csv("data/test_Data.csv", sep=";")

target = "Progression_Status"

X_test_raw = test_data.drop(columns=[target])
y_test = test_data[target]

X_test_raw = pd.get_dummies(X_test_raw)

log("Test loaded successfully")


# =========================================================
# ALIGN FEATURES
# =========================================================
def align_features(X, model):
    expected_cols = model.feature_names_in_
    X_aligned = X.copy()

    for col in expected_cols:
        if col not in X_aligned.columns:
            X_aligned[col] = 0

    return X_aligned[expected_cols]


# =========================================================
# SAVE METRICS
# =========================================================
def save_dashboard_metrics(round_num, acc, sens, spec, prec, f1, auc):

    metrics = {
        "round": round_num,
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "auc": float(auc)
    }

    with open("dashboard/global_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


# =========================================================
# STRATEGY
# =========================================================
class FedTreeStrategy(fl.server.strategy.FedAvg):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_training = False

    # =========================
    # INIT
    # =========================
    def initialize_parameters(self, client_manager):

        log("[SERVER] Sending initial model...")

        config = {
            "model": GM,
            "K": K_VALUE,
            "params": GM.get_params()
        }

        return fl.common.ndarrays_to_parameters(
            [pickle.dumps(config)]
        )

    # =========================
    # FIT AGGREGATION
    # =========================
    def aggregate_fit(self, server_round, results, failures):

        global GM

        new_trees = []

        log(f"\n===== ROUND {server_round} =====")

        for i, (_, fit_res) in enumerate(results):

            bij = fit_res.metrics.get("bij")
            tij_bytes = fit_res.metrics.get("tij")

            if bij:
                log(f"Client {i+1}: already up to date")

            elif tij_bytes:
                trees = pickle.loads(tij_bytes)
                new_trees.extend(trees)

                log(f"Client {i+1}: {len(trees)} trees received")

            else:
                log(f"Client {i+1}: invalid response")

        if len(new_trees) > 0:

            GM.estimators_.extend(new_trees)
            GM.n_estimators = len(GM.estimators_)

            log(f"[SERVER] +{len(new_trees)} trees added")
            log(f"[SERVER] Model size: {GM.n_estimators}")

        else:
            log("[SERVER] No updates received → early stop")
            self.stop_training = True

        config_next = {
            "model": GM,
            "K": K_VALUE,
            "params": GM.get_params()
        }

        return (
            fl.common.ndarrays_to_parameters([pickle.dumps(config_next)]),
            {}
        )

    # =========================
    # EVALUATION
    # =========================
    def evaluate_global_model(self, server_round):

        global GM

        log("\n===================================")
        log(f"GLOBAL MODEL ROUND {server_round}")
        log("===================================")

        X_test = align_features(X_test_raw, GM)

        y_pred = GM.predict(X_test)
        y_prob = GM.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        sens = recall_score(y_test, y_pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        prec = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        log(f"Accuracy: {acc:.4f}")
        log(f"Sensitivity: {sens:.4f}")
        log(f"Specificity: {spec:.4f}")
        log(f"Precision: {prec:.4f}")
        log(f"F1: {f1:.4f}")
        log(f"AUC: {auc:.4f}")

        save_dashboard_metrics(server_round, acc, sens, spec, prec, f1, auc)

    # =========================
    # EVALUATE AGGREGATION
    # =========================
    def aggregate_evaluate(self, server_round, results, failures):

        loss, _ = super().aggregate_evaluate(server_round, results, failures)

        self.evaluate_global_model(server_round)

        return loss, {}


# =========================================================
# START SERVER
# =========================================================
if __name__ == "__main__":

    strategy = FedTreeStrategy(
        min_fit_clients=NB_CLIENTS,
        min_available_clients=NB_CLIENTS,
        min_evaluate_clients=NB_CLIENTS
    )

    try:
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=15),
            strategy=strategy
        )

    except Exception as e:
        log(f"ERROR: {e}")

    log("\nTRAINING FINISHED")
    os._exit(0)