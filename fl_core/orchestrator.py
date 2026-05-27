import pandas as pd
import numpy as np
import os


class Orchestrator:

    def __init__(self, dataset_path="data/735_Data.csv"):
        self.dataset_path = dataset_path
        self.df = None

    # =========================
    # LOAD DATASET
    # =========================
    def load_dataset(self):

        print("[ORCH] Loading dataset...")

        # 🔥 IMPORTANT: auto detect separator
        self.df = pd.read_csv(self.dataset_path, sep=None, engine="python")

        print(f"[ORCH] Dataset loaded: {len(self.df)} rows")

    # =========================
    # SPLIT CLIENTS
    # =========================
    def split_clients(self, num_clients=3):

        print(f"[ORCH] Splitting dataset into {num_clients} clients...")

        if self.df is None:
            raise ValueError("Dataset not loaded")

        df_shuffled = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

        splits = np.array_split(df_shuffled, num_clients)

        output_dir = "orchestrator/client_data"
        os.makedirs(output_dir, exist_ok=True)

        for i, client_df in enumerate(splits):

            path = f"{output_dir}/client{i+1}.csv"

            # 🔥 IMPORTANT: force clean format
            client_df.to_csv(path, index=False, sep=",")

            print(f"[ORCH] Client {i+1} saved -> {path}")

        print("[ORCH] Splitting finished ✔")

    # =========================
    # TEST SET
    # =========================
    def create_test_set(self, test_size=0.2):

        print("[ORCH] Creating global test set...")

        df = self.df.sample(frac=1, random_state=42)

        split_idx = int(len(df) * (1 - test_size))

        test_df = df.iloc[split_idx:]

        os.makedirs("data", exist_ok=True)

        test_df.to_csv(
            "data/test_Data.csv",
            index=False,
            sep=","   # 🔥 IMPORTANT FIX
        )

        print("[ORCH] Test set saved ✔")


# =========================
# RUN
# =========================
if __name__ == "__main__":

    orch = Orchestrator()

    orch.load_dataset()
    orch.split_clients(3)
    orch.create_test_set()