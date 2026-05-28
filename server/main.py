from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import os
import sys
import json

from fl_core.orchestrator import Orchestrator

app = FastAPI()

SERVER_PROCESS = None
CLIENTS = []


# =========================================
# REQUEST MODEL
# =========================================
class FLConfig(BaseModel):
    num_clients: int
    k_value: int
    model_choice: str
    dataset_path: str   # 👈 important (flexible)
    split_strategy: str # kept only for logging/UI (optionnel)


# =========================================
# START FL
# =========================================
@app.post("/start_fl")
def start_fl(config: FLConfig):

    global SERVER_PROCESS, CLIENTS

    # =====================================
    # STOP OLD PROCESS
    # =====================================
    if SERVER_PROCESS:
        SERVER_PROCESS.kill()

    for c in CLIENTS:
        c.kill()

    CLIENTS = []

    # =====================================
    # ORCHESTRATOR (CLIENT DATA ONLY)
    # =====================================
    orch = Orchestrator()

    orch.load_dataset(config.dataset_path)

    orch.split_clients(
        num_clients=config.num_clients
    )

    # =====================================
    # SERVER CONFIG (IMPORTANT)
    # =====================================
    server_config = {
        "model": f"models/{config.model_choice}",  # pretrained model
        "k": config.k_value,
        "clients": config.num_clients
    }

    os.makedirs("server", exist_ok=True)

    with open("server/config.json", "w") as f:
        json.dump(server_config, f, indent=4)

    # =====================================
    # START SERVER
    # =====================================
    SERVER_PROCESS = subprocess.Popen([
        sys.executable,
        "server/fl_server.py"
    ])

    # =====================================
    # START CLIENTS
    # =====================================
    for i in range(1, config.num_clients + 1):

        p = subprocess.Popen([
            sys.executable,
            "clients/client.py",
            str(i)
        ])

        CLIENTS.append(p)

    return {
        "status": "FL started successfully",
        "clients": config.num_clients,
        "k": config.k_value,
        "model": config.model_choice,
        "dataset": config.dataset_path
    }


# =========================================
# STOP FL
# =========================================
@app.post("/stop_fl")
def stop_fl():

    global SERVER_PROCESS, CLIENTS

    if SERVER_PROCESS:
        SERVER_PROCESS.kill()

    for c in CLIENTS:
        c.kill()

    CLIENTS = []

    return {
        "status": "FL stopped"
    }