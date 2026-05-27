from fastapi import FastAPI
import subprocess
import os
import sys
import json

from fl_core.orchestrator import Orchestrator

app = FastAPI()

SERVER_PROCESS=None
CLIENTS=[]


@app.post("/start_fl")
def start_fl():

    global SERVER_PROCESS
    global CLIENTS

    # -------- ORCHESTRATOR --------

    orch=Orchestrator()

    orch.load_dataset()

    orch.split_clients(
        num_clients=3
    )

    # -------- SERVER CONFIG --------

    config={

        "model":
        "models/random_forest_model_tcga+emtab500.pkl",

        "k":10,

        "clients":3

    }

    with open(
        "server/config.json",
        "w"
    ) as f:

        json.dump(
            config,
            f,
            indent=4
        )

    # -------- START SERVER --------

    SERVER_PROCESS= subprocess.Popen(

        [sys.executable,
         "server/fl_server.py"]

    )

    # -------- START CLIENTS --------

    for i in range(1,4):

        p=subprocess.Popen(

            [sys.executable,
             f"clients/client{i}.py"]

        )

        CLIENTS.append(p)

    return{

        "status":
        "FL started"
    }


@app.post("/stop_fl")
def stop_fl():

    global SERVER_PROCESS
    global CLIENTS

    if SERVER_PROCESS:

        SERVER_PROCESS.kill()

    for c in CLIENTS:

        c.kill()

    CLIENTS=[]

    return{

        "status":
        "stopped"
    }