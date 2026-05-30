import pandas as pd
import sys
import os

def split_client_data(df, num_clients, prefix=""):
    """
    Découpe dynamiquement le sous-dataset fourni (367 ou 368) en fonction du nombre de clients.
    L'ordre original du premier script est préservé (pas de double shuffle).
    """
    output_dir = "orchestrator/client_data"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[{prefix}] Dispatching into {num_clients} client files in '{output_dir}'...")

    if num_clients == 2:
        df.iloc[:147].to_csv(f"{output_dir}/client1.csv", index=False, sep=";")
        df.iloc[147:].to_csv(f"{output_dir}/client2.csv", index=False, sep=";")
        
    elif num_clients == 3:
        df.iloc[:110].to_csv(f"{output_dir}/client1.csv", index=False, sep=";")
        df.iloc[110:276].to_csv(f"{output_dir}/client2.csv", index=False, sep=";")
        df.iloc[276:].to_csv(f"{output_dir}/client3.csv", index=False, sep=";")
        
    elif num_clients == 4:
        df.iloc[:110].to_csv(f"{output_dir}/client1.csv", index=False, sep=";")
        df.iloc[110:184].to_csv(f"{output_dir}/client2.csv", index=False, sep=";")
        df.iloc[184:239].to_csv(f"{output_dir}/client3.csv", index=False, sep=";")
        df.iloc[239:].to_csv(f"{output_dir}/client4.csv", index=False, sep=";")
        
    elif num_clients == 5:
        df.iloc[:74].to_csv(f"{output_dir}/client1.csv", index=False, sep=";")
        df.iloc[74:184].to_csv(f"{output_dir}/client2.csv", index=False, sep=";")
        df.iloc[184:239].to_csv(f"{output_dir}/client3.csv", index=False, sep=";")
        df.iloc[239:331].to_csv(f"{output_dir}/client4.csv", index=False, sep=";")
        df.iloc[331:].to_csv(f"{output_dir}/client5.csv", index=False, sep=";")
    else:
        raise ValueError("num_clients doit être compris entre 2 et 5")


if __name__ == "__main__":
    # =====================================================
    # 1. RÉCUPÉRATION DES ARGUMENTS
    # =====================================================
    if len(sys.argv) < 3:
        print("Usage: python split_367_368.py [num_clients] [dataset_name]")
        sys.exit(1)

    num_clients = int(sys.argv[1])
    dataset_choisi = sys.argv[2]

    print(f"\n[SCRIPT 367/368] Arguments reçus -> Clients: {num_clients} | Dataset ciblé: {dataset_choisi}")

    # =====================================================
    # 2. CHARGEMENT ET PRE-SPLIT DU DATASET SOURCE
    # =====================================================
    source_file = "data/735_Data.csv"
    if not os.path.exists(source_file):
        print(f"Erreur : Le fichier source '{source_file}' est introuvable.")
        sys.exit(1)

    df_global = pd.read_csv(source_file, sep=None, engine="python")
    
    # Mélange initial global avec random_state=42 (conforme à tes premiers tests 367/368)
    df_global = df_global.sample(frac=1, random_state=42).reset_index(drop=True)

    # Séparation stricte 367 / 368
    df_367 = df_global.iloc[:367]
    df_368 = df_global.iloc[367:]

    # Sauvegarde des fichiers intermédiaires globaux à la racine (requis par l'API)
    df_367.to_csv("367_Data.csv", index=False, sep=";")
    df_368.to_csv("368_Data.csv", index=False, sep=";")
    print("[SCRIPT 367/368] Fichiers globaux '367_Data.csv' et '368_Data.csv' sauvegardés à la racine.")

    # =====================================================
    # 3. GÉNÉRATION DYNAMIQUE DES CONFIGURATIONS CLIENTS
    # =====================================================
    # CORRECTION : On aiguille correctement vers le bon dataframe selon le choix de l'utilisateur
    if "368" in dataset_choisi:
        split_client_data(df_368, num_clients, prefix="SPLIT 368")
    else:
        split_client_data(df_367, num_clients, prefix="SPLIT 367")

    print("[SCRIPT 367/368] Processus de split terminé avec succès ✔\n")