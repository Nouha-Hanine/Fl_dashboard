import pandas as pd
import sys
import os

if __name__ == "__main__":
    # 0. Récupération des variables envoyées par le bouton de ton FastAPI
    num_clients = int(sys.argv[1])
    dataset_path = sys.argv[2]  # Reçoit "data/735_Data.csv" depuis FastAPI
    target_block = sys.argv[3]
    output_dir = "orchestrator/client_data"
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================
    # ÉTAPE 1 : DIVISION DU FICHIER GLOBAL DE 735 LIGNES (RS=10)
    # =========================================================
    df_735 = pd.read_csv(dataset_path)
    df_735_shuffled = df_735.sample(frac=1, random_state=10).reset_index(drop=True)

    # Split the shuffled dataframe
    df_500_block = df_735_shuffled.iloc[:500].reset_index(drop=True)
    df_235_block = df_735_shuffled.iloc[500:].reset_index(drop=True)

    # Save the resulting dataframes to new CSV files à la racine
    df_500_block.to_csv('500_Data.csv', index=False)
    df_235_block.to_csv('235_Data.csv', index=False)

    print(f"File '500_Data.csv' created with {len(df_500_block)} rows.")
    print(f"File '235_Data.csv' created with {len(df_235_block)} rows.")
    print(f"Target block selected by dashboard: {target_block}")
    # =========================================================
    # OPTION A : LOGIQUE DE DÉCOUPAGE POUR LE BLOC 235
    # =========================================================
    if "235" in target_block:
        df_235 = pd.read_csv('235_Data.csv')
        df_shuffled = df_235.sample(frac=1, random_state=42).reset_index(drop=True)

        # --- CONFIGURATION 1: 2 Parts (40%, 60%) ---
        if num_clients == 2:
            split2_40 = df_shuffled.iloc[:94]
            split2_60 = df_shuffled.iloc[94:]

            split2_40.to_csv('config2_235_40.csv', index=False)
            split2_60.to_csv('config2_235_60.csv', index=False)
            
            split2_40.to_csv(os.path.join(output_dir, 'client1.csv'), index=False)
            split2_60.to_csv(os.path.join(output_dir, 'client2.csv'), index=False)
            print("Saved 2-way split files for 235.")

        # --- CONFIGURATION 2: 3 Parts (30%, 45%, 25%) ---
        elif num_clients == 3:
            split3_30 = df_shuffled.iloc[:71]
            split3_45 = df_shuffled.iloc[71:177]
            split3_25 = df_shuffled.iloc[177:]

            split3_30.to_csv('config3_235_30_.csv', index=False)
            split3_45.to_csv('config3_235_45.csv', index=False)
            split3_25.to_csv('config3_235_25.csv', index=False)
            
            split3_30.to_csv(os.path.join(output_dir, 'client1.csv'), index=False)
            split3_45.to_csv(os.path.join(output_dir, 'client2.csv'), index=False)
            split3_25.to_csv(os.path.join(output_dir, 'client3.csv'), index=False)
            print("Saved 3-way split files for 235.")

        # --- CONFIGURATION 3: 4 Parts (30%, 20%, 15%, 35%) ---
        elif num_clients == 4:
            split4_30 = df_shuffled.iloc[:71]
            split4_20 = df_shuffled.iloc[71:118]
            split4_15 = df_shuffled.iloc[118:153]
            split4_35 = df_shuffled.iloc[153:]

            split4_30.to_csv('config4_235_30.csv', index=False)
            split4_20.to_csv('config4_235_20.csv', index=False)
            split4_15.to_csv('config4_235_15.csv', index=False)
            split4_35.to_csv('config4_235_35.csv', index=False)
            
            split4_30.to_csv(os.path.join(output_dir, 'client1.csv'), index=False)
            split4_20.to_csv(os.path.join(output_dir, 'client2.csv'), index=False)
            split4_15.to_csv(os.path.join(output_dir, 'client3.csv'), index=False)
            split4_35.to_csv(os.path.join(output_dir, 'client4.csv'), index=False)
            print("Saved 4-way split files for 235.")

        # --- CONFIGURATION 4: 5 Parts (20%, 30%, 15%, 25%, 10%) ---
        elif num_clients == 5:
            split5_235_20 = df_shuffled.iloc[:47]
            split5_235_30 = df_shuffled.iloc[47:118]
            split5_235_15 = df_shuffled.iloc[118:153]
            split5_235_25 = df_shuffled.iloc[153:212]
            split5_235_10 = df_shuffled.iloc[212:]

            split5_235_20.to_csv('config5_235_20.csv', index=False)
            split5_235_30.to_csv('config5_235_30.csv', index=False)
            split5_235_15.to_csv('config5_235_15.csv', index=False)
            split5_235_25.to_csv('config5_235_25.csv', index=False)
            split5_235_10.to_csv('config5_235_10.csv', index=False)
            
            split5_235_20.to_csv(os.path.join(output_dir, 'client1.csv'), index=False)
            split5_235_30.to_csv(os.path.join(output_dir, 'client2.csv'), index=False)
            split5_235_15.to_csv(os.path.join(output_dir, 'client3.csv'), index=False)
            split5_235_25.to_csv(os.path.join(output_dir, 'client4.csv'), index=False)
            split5_235_10.to_csv(os.path.join(output_dir, 'client5.csv'), index=False)
            print("Saved 235_Data Config 5 files.")

    # =========================================================
    # OPTION B : LOGIQUE DE DÉCOUPAGE POUR LE BLOC 500
    # =========================================================
    else:
        df_500 = pd.read_csv('500_Data.csv')
        df_shuffled = df_500.sample(frac=1, random_state=42).reset_index(drop=True)

        # --- CONFIGURATION 1: 2 Parts (40%, 60%) ---
        if num_clients == 2:
            split2_40 = df_shuffled.iloc[:200]
            split2_60 = df_shuffled.iloc[200:]

            split2_40.to_csv('config2_500_40.csv', index=False)
            split2_60.to_csv('config2_500_60.csv', index=False)
            
            split2_40.to_csv(os.path.join(output_dir, 'client1.csv'), index=False)
            split2_60.to_csv(os.path.join(output_dir, 'client2.csv'), index=False)
            print("Saved 2-way split files for 500.")

        # --- CONFIGURATION 2: 3 Parts (30%, 45%, 25%) ---
        elif num_clients == 3:
            split3_30 = df_shuffled.iloc[:150]
            split3_45 = df_shuffled.iloc[150:375]
            split3_25 = df_shuffled.iloc[375:]

            split3_30.to_csv('config3_500_30.csv', index=False)
            split3_45.to_csv('config3_500_45.csv', index=False)
            split3_25.to_csv('config3_500_25.csv', index=False)
            
            split3_30.to_csv(os.path.join(output_dir, 'client1.csv'), index=False)
            split3_45.to_csv(os.path.join(output_dir, 'client2.csv'), index=False)
            split3_25.to_csv(os.path.join(output_dir, 'client3.csv'), index=False)
            print("Saved 3-way split files for 500.")

        # --- CONFIGURATION 3: 4 Parts (30%, 20%, 15%, 35%) ---
        elif num_clients == 4:
            split4_30 = df_shuffled.iloc[:150]
            split4_20 = df_shuffled.iloc[150:250]
            split4_15 = df_shuffled.iloc[250:325]
            split4_35 = df_shuffled.iloc[325:]

            split4_30.to_csv('config4_500_30.csv', index=False)
            split4_20.to_csv('config4_500_20.csv', index=False)
            split4_15.to_csv('config4_500_15.csv', index=False)
            split4_35.to_csv('config4_500_35.csv', index=False)
            
            split4_30.to_csv(os.path.join(output_dir, 'client1.csv'), index=False)
            split4_20.to_csv(os.path.join(output_dir, 'client2.csv'), index=False)
            split4_15.to_csv(os.path.join(output_dir, 'client3.csv'), index=False)
            split4_35.to_csv(os.path.join(output_dir, 'client4.csv'), index=False)
            print("Saved 4-way split files for 500.")

        # --- CONFIGURATION 4: 5 Parts (20%, 30%, 15%, 25%, 10%) ---
        elif num_clients == 5:
            split5_500_20 = df_shuffled.iloc[:100]
            split5_500_30 = df_shuffled.iloc[100:250]
            split5_500_15 = df_shuffled.iloc[250:325]
            split5_500_25 = df_shuffled.iloc[325:450]
            split5_500_10 = df_shuffled.iloc[450:]

            split5_500_20.to_csv('config5_500_20.csv', index=False)
            split5_500_30.to_csv('config5_500_30.csv', index=False)
            split5_500_15.to_csv('config5_500_15.csv', index=False)
            split5_500_25.to_csv('config5_500_25.csv', index=False)
            split5_500_10.to_csv('config5_500_10.csv', index=False)
            
            split5_500_20.to_csv(os.path.join(output_dir, 'client1.csv'), index=False)
            split5_500_30.to_csv(os.path.join(output_dir, 'client2.csv'), index=False)
            split5_500_15.to_csv(os.path.join(output_dir, 'client3.csv'), index=False)
            split5_500_25.to_csv(os.path.join(output_dir, 'client4.csv'), index=False)
            split5_500_10.to_csv(os.path.join(output_dir, 'client5.csv'), index=False)
            print("Saved 500_Data Config 5 files.")