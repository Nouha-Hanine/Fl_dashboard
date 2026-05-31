from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os
import sys
import json
import threading

app = FastAPI()

SERVER_PROCESS = None
CLIENTS = []
# Lock to prevent concurrent overlapping training sessions
TRAINING_LOCK = threading.Lock() 

# =========================================================
# REQUEST MODEL
# =========================================================
class FLConfig(BaseModel):
    num_clients: int
    k_value: int
    model_choice: str
    dataset: str
    split_strategy: str

# =========================================================
# START FL
# =========================================================
@app.post("/start_fl")
def start_fl(config: FLConfig):
    global SERVER_PROCESS, CLIENTS

    with TRAINING_LOCK:
        # =====================================================
        # CLEANUP PREVIOUS PROCESSES
        # =====================================================
        if SERVER_PROCESS:
            try:
                SERVER_PROCESS.terminate()
                SERVER_PROCESS.wait(timeout=5)
            except Exception:
                SERVER_PROCESS.kill()

        for c in CLIENTS:
            try:
                c.terminate()
                c.wait(timeout=2)
            except Exception:
                c.kill()
        
        CLIENTS = []

        # =====================================================
        # BASE DIRECTORIES CREATION
        # =====================================================
        os.makedirs("orchestrator/client_data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("server", exist_ok=True)

        # Reset the server log file cleanly for the new session
        with open("logs/server.log", "w", encoding="utf-8") as f:
            f.write("=== NEW FEDERATED LEARNING SESSION ===\n")

        # =====================================================
        # STEP 1: SAVE CONFIGURATION (Priority for external scripts)
        # =====================================================
        server_config = {
            "model": f"models/{config.model_choice}",
            "k": config.k_value,
            "clients": config.num_clients,
            "dataset": config.dataset,
            "split_strategy": config.split_strategy
        }

        with open("server/config.json", "w", encoding="utf-8") as f:
            json.dump(server_config, f, indent=4)

        # =====================================================
        # STEP 2: DATA SPLIT VIA COMMAND LINE ARGUMENTS (CORRIGÉ)
        # =====================================================
        try:
            if config.split_strategy == "500 / 235":
                subprocess.run([
                    sys.executable, "scripts/split_500_235.py", 
                    str(config.num_clients), 
                    "data/735_Data.csv",     # sys.argv[2] -> Chemin réel de ton fichier source
                    config.dataset           # sys.argv[3] -> Bloc cible dynamique ("235" ou "500")
                ], check=True)
            elif config.split_strategy == "367 / 368":
                subprocess.run([
                    sys.executable, "scripts/split_367_368.py", 
                    str(config.num_clients), 
                    "data/735_Data.csv",     # Alignement du chemin source
                    config.dataset           # Envoi de la configuration cible
                ], check=True)
            else:
                raise HTTPException(status_code=400, detail="Invalid split strategy")
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"Data split script error: {e}")

        # =====================================================
        # STEP 3: VERIFY GENERATED CLIENT FILES
        # =====================================================
        for i in range(config.num_clients):
            expected_file = f"orchestrator/client_data/client{i+1}.csv"
            if not os.path.exists(expected_file):
                raise HTTPException(
                    status_code=500, 
                    detail=f"The data split script failed to generate: {expected_file}"
                )

        # =====================================================
        # STEP 4: START FL SERVER AND CLIENT SUBPROCESSES
        # =====================================================
        try:
            log_output = open("logs/server.log", "a", encoding="utf-8")
            
            # Start FL Server
            SERVER_PROCESS = subprocess.Popen(
                [sys.executable, "server/fl_server.py"],
                stdout=log_output,
                stderr=log_output,
                text=True
            )

            # Start FL Clients
            for i in range(config.num_clients):
                p = subprocess.Popen(
                    [sys.executable, "clients/client.py", str(i + 1)],
                    stdout=log_output,
                    stderr=log_output,
                    text=True
                )
                CLIENTS.append(p)
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Subprocess execution error: {e}")

        return {
            "status": "FL started successfully",
            "clients": config.num_clients,
            "k": config.k_value
        }

# =========================================================
# STOP FL
# =========================================================
@app.post("/stop_fl")
def stop_fl():
    global SERVER_PROCESS, CLIENTS

    with TRAINING_LOCK:
        if SERVER_PROCESS:
            SERVER_PROCESS.terminate()
            SERVER_PROCESS.wait()
            SERVER_PROCESS = None

        for c in CLIENTS:
            c.terminate()
            c.wait()
        
        CLIENTS = []

        return {"status": "FL stopped successfully"}