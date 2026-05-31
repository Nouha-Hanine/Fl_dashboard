import pandas as pd
import sys
import os

if __name__ == "__main__":
    # 0. Récupération des variables envoyées par le bouton de ton FastAPI
    num_clients = int(sys.argv[1])
    dataset_path = sys.argv[2]  # Reçoit "data/735_Data.csv" depuis FastAPI

    output_dir = "orchestrator/client_data"
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================
    # ÉTAPE 1 : DIVISION DU FICHIER GLOBAL DE 735 LIGNES
    # =========================================================
    df_735 = pd.read_csv(dataset_path)
    df_735_shuffled = df_735.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split the shuffled dataframe
    df_367 = df_735_shuffled.iloc[:367].reset_index(drop=True)
    df_368 = df_735_shuffled.iloc[367:].reset_index(drop=True)

    # Save the resulting dataframes to new CSV files à la racine
    df_367.to_csv('367_Data.csv', index=False)
    df_368.to_csv('368_Data.csv', index=False)

    print(f"File '367_Data.csv' created with {len(df_367)} rows.")
    print(f"File '368_Data.csv' created with {len(df_368)} rows.")

    # =========================================================
    # ÉTAPE 2 : DEUXIÈME SHUFFLE ET SÉPARATION CLIENTS (368 LIGNES)
    # =========================================================
    # 1. Load the dataset (on reprend le fichier fraîchement créé)
    df_368_loaded = pd.read_csv('368_Data.csv')

    # 2. Remix (shuffle) the data before splitting
    df_shuffled = df_368_loaded.sample(frac=1, random_state=42).reset_index(drop=True)

    # ==========================================
    # CONFIGURATION 1: 2 Parts (40%, 60%)
    # ==========================================
    if num_clients == 2:
        split2_40 = df_shuffled.iloc[:147]
        split2_60 = df_shuffled.iloc[147:]

        # Sauvegarde pour tes archives locales
        split2_40.to_csv('config2_40_368.csv', index=False)
        split2_60.to_csv('config2_60_368.csv', index=False)
        
        # Envoi direct à l'orchestrator pour FastAPI
        split2_40.to_csv(os.path.join(output_dir, 'client1.csv'), index=False)
        split2_60.to_csv(os.path.join(output_dir, 'client2.csv'), index=False)
        print("Saved 2-way split files.")

    # ==========================================
    # CONFIGURATION 2: 3 Parts (30%, 45%, 25%)
    # ==========================================
    elif num_clients == 3:
        split3_30 = df_shuffled.iloc[:110]
        split3_45 = df_shuffled.iloc[110:276]  # 110 + 166 = 276
        split3_25 = df_shuffled.iloc[276:]

        # Sauvegarde pour tes archives locales
        split3_30.to_csv('config3_30_368.csv', index=False)
        split3_45.to_csv('config3_45_368.csv', index=False)
        split3_25.to_csv('config3_25_368.csv', index=False)
        
        # Envoi direct à l'orchestrator pour FastAPI
        split3_30.to_csv(os.path.join(output_dir, 'client1.csv'), index=False)
        split3_45.to_csv(os.path.join(output_dir, 'client2.csv'), index=False)
        split3_25.to_csv(os.path.join(output_dir, 'client3.csv'), index=False)
        print("Saved 3-way split files.")

    # ==========================================
    # CONFIGURATION 3: 4 Parts (30%, 20%, 15%, 35%)
    # ==========================================
    elif num_clients == 4:
        split4_30 = df_shuffled.iloc[:110]
        split4_20 = df_shuffled.iloc[110:184]  # 110 + 74 = 184
        split4_15 = df_shuffled.iloc[184:239]  # 184 + 55 = 239
        split4_35 = df_shuffled.iloc[239:]     # Remaining 129 rows

        # Sauvegarde pour tes archives locales
        split4_30.to_csv('config4_30_368.csv', index=False)
        split4_20.to_csv('config4_20_368.csv', index=False)
        split4_15.to_csv('config4_15_368.csv', index=False)
        split4_35.to_csv('config4_35_368.csv', index=False)
        
        # Envoi direct à l'orchestrator pour FastAPI
        split4_30.to_csv(os.path.join(output_dir, 'client1.csv'), index=False)
        split4_20.to_csv(os.path.join(output_dir, 'client2.csv'), index=False)
        split4_15.to_csv(os.path.join(output_dir, 'client3.csv'), index=False)
        split4_35.to_csv(os.path.join(output_dir, 'client4.csv'), index=False)
        print("Saved 4-way split files.")

    # ==========================================
    # CONFIGURATION 4: 5 Parts (20%, 30%, 15%, 25%, 10%)
    # ==========================================
    elif num_clients == 5:
        split5_368_20 = df_shuffled.iloc[:74]
        split5_368_30 = df_shuffled.iloc[74:184]
        split5_368_15 = df_shuffled.iloc[184:239]
        split5_368_25 = df_shuffled.iloc[239:331]
        split5_368_10 = df_shuffled.iloc[331:]

        # Sauvegarde pour tes archives locales
        split5_368_20.to_csv('config5_368_20.csv', index=False)
        split5_368_30.to_csv('config5_368_30.csv', index=False)
        split5_368_15.to_csv('config5_368_15.csv', index=False)
        split5_368_25.to_csv('config5_368_25.csv', index=False)
        split5_368_10.to_csv('config5_368_10.csv', index=False)
        
        # Envoi direct à l'orchestrator pour FastAPI
        split5_368_20.to_csv(os.path.join(output_dir, 'client1.csv'), index=False)
        split5_368_30.to_csv(os.path.join(output_dir, 'client2.csv'), index=False)
        split5_368_15.to_csv(os.path.join(output_dir, 'client3.csv'), index=False)
        split5_368_25.to_csv(os.path.join(output_dir, 'client4.csv'), index=False)
        split5_368_10.to_csv(os.path.join(output_dir, 'client5.csv'), index=False)
        print("Saved 368_Data Config 5 files.")