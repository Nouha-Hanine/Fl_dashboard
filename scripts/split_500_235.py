import pandas as pd
import sys
import os

def split_500_data(df, num_clients):
    """
    Découpe dynamiquement le sous-dataset 500 en fonction du nombre de clients.
    """
    output_dir = "orchestrator/client_data"
    os.makedirs(output_dir, exist_ok=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"[SPLIT 500] Dispatching into {num_clients} client files...")

    if num_clients == 2:
        df.iloc[:200].to_csv(f"{output_dir}/client1.csv", index=False, sep=";")
        df.iloc[200:].to_csv(f"{output_dir}/client2.csv", index=False, sep=";")
    elif num_clients == 3:
        df.iloc[:150].to_csv(f"{output_dir}/client1.csv", index=False, sep=";")
        df.iloc[150:375].to_csv(f"{output_dir}/client2.csv", index=False, sep=";")
        df.iloc[375:].to_csv(f"{output_dir}/client3.csv", index=False, sep=";")
    elif num_clients == 4:
        df.iloc[:150].to_csv(f"{output_dir}/client1.csv", index=False, sep=";")
        df.iloc[150:250].to_csv(f"{output_dir}/client2.csv", index=False, sep=";")
        df.iloc[250:325].to_csv(f"{output_dir}/client3.csv", index=False, sep=";")
        df.iloc[325:].to_csv(f"{output_dir}/client4.csv", index=False, sep=";")
    elif num_clients == 5:
        df.iloc[:100].to_csv(f"{output_dir}/client1.csv", index=False, sep=";")
        df.iloc[100:250].to_csv(f"{output_dir}/client2.csv", index=False, sep=";")
        df.iloc[250:325].to_csv(f"{output_dir}/client3.csv", index=False, sep=";")
        df.iloc[325:450].to_csv(f"{output_dir}/client4.csv", index=False, sep=";")
        df.iloc[450:].to_csv(f"{output_dir}/client5.csv", index=False, sep=";")
    else:
        raise ValueError("num_clients doit être compris entre 2 et 5")


def split_235_data(df, num_clients):
    """
    Découpe le sous-dataset 235 selon les ratios asymétriques spécifiés.
    """
    output_dir = "orchestrator/client_data"
    os.makedirs(output_dir, exist_ok=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    
    print(f"[SPLIT 235] Dispatching into {num_clients} client files...")

    if num_clients == 2: 
        ratios = [0.40, 0.60]
    elif num_clients == 3: 
        ratios = [0.30, 0.45, 0.25]
    elif num_clients == 4: 
        ratios = [0.30, 0.20, 0.15, 0.35]
    elif num_clients == 5: 
        ratios = [0.20, 0.30, 0.15, 0.25, 0.10]
    else:
        raise ValueError("num_clients doit être compris entre 2 et 5")
    
    start = 0
    for i, ratio in enumerate(ratios):
        if i == len(ratios) - 1:
            part = df.iloc[start:]
        else:
            end = start + int(n * ratio)
            part = df.iloc[start:end]
            
        part.to_csv(f"{output_dir}/client{i+1}.csv", index=False)
        print(f"[SPLIT 235] Client {i+1} saved -> {len(part)} rows")
        start = end


if __name__ == "__main__":
    # =====================================================
    # 1. RÉCUPÉRATION DES ARGUMENTS DEPUIS L'API FASTAPI
    # =====================================================
    if len(sys.argv) < 3:
        print("Usage: python split_500_235.py [num_clients] [dataset_name]")
        sys.exit(1)

    num_clients = int(sys.argv[1])
    dataset_choisi = sys.argv[2]

    print(f"\n[SCRIPT 500/235] Arguments reçus -> Clients: {num_clients} | Dataset ciblé: {dataset_choisi}")

    # =====================================================
    # 2. CHARGEMENT ET PRE-SPLIT DU DATASET SOURCE
    # =====================================================
    source_file = "data/735_Data.csv"
    if not os.path.exists(source_file):
        print(f"Erreur : Le fichier source '{source_file}' est introuvable.")
        sys.exit(1)

    df_global = pd.read_csv(source_file, sep=None, engine="python")
    
    # Mélange initial du dataset complet
    df_global = df_global.sample(frac=1, random_state=42).reset_index(drop=True)

    # Séparation stricte 500 / 235
    df_500 = df_global.iloc[:500]
    df_235 = df_global.iloc[500:]

    # Sauvegarde des fichiers intermédiaires globaux à la racine (requis par l'API)
    df_500.to_csv("500_Data.csv", index=False, sep=";")
    df_235.to_csv("235_Data.csv", index=False, sep=";")
    print("[SCRIPT 500/235] Fichiers globaux '500_Data.csv' et '235_Data.csv' sauvegardés à la racine.")

    # =====================================================
    # 3. GÉNÉRATION DYNAMIQUE DES CONFIGURATIONS CLIENTS
    # =====================================================
    if "500" in dataset_choisi:
        split_500_data(df_500, num_clients)
    else:
        split_235_data(df_235, num_clients)

    print("[SCRIPT 500/235] Processus de split terminé avec succès ✔\n")