import pandas as pd
import numpy as np
import os


class Orchestrator:

    def __init__(self):
        self.df = None

    # =====================================================
    # LOAD DATASET
    # =====================================================
    def load_dataset(self, dataset_path):

        print("\n[ORCH] Loading dataset...")

        self.df = pd.read_csv(dataset_path, sep=";", engine="python")

        print(f"[ORCH] Dataset loaded: {len(self.df)} rows")

        # sécurité
        if "Progression_Status" not in self.df.columns:
            raise ValueError("Target column 'Progression_Status' not found!")

    # =====================================================
    # SPLIT ONLY FOR CLIENTS
    # =====================================================
    def split_clients(self, num_clients=3):

        print(f"\n[ORCH] Splitting dataset into {num_clients} clients...")

        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_dataset first.")

        # shuffle dataset
        df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

        # split
        splits = np.array_split(df, num_clients)

        output_dir = "orchestrator/client_data"
        os.makedirs(output_dir, exist_ok=True)

        # save each client file
        for i, client_df in enumerate(splits):

            path = f"{output_dir}/client{i+1}.csv"

            # IMPORTANT: keep same format everywhere
            client_df.to_csv(path, index=False, sep=";")

            print(f"[ORCH] Client {i+1} saved -> {path} ({len(client_df)} rows)")

        print("[ORCH] Client split finished ✔")

        return splits