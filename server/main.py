from fastapi import FastAPI
import subprocess
import os
import sys

app = FastAPI()

FL_PROCESS = None


@app.post("/start_fl")
def start_fl():
    global FL_PROCESS

    if FL_PROCESS is None:

        FL_PROCESS = subprocess.Popen(
            [sys.executable, "server/fl_server.py"],
            cwd=os.getcwd()
        )

        return {"status": "FL started"}

    return {"status": "already running"}


@app.post("/stop_fl")
def stop_fl():
    global FL_PROCESS

    if FL_PROCESS:

        FL_PROCESS.terminate()
        FL_PROCESS = None

        return {"status": "FL stopped"}

    return {"status": "not running"}