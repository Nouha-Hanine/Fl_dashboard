import pandas as pd
import sys
import os

# =====================================================
# INPUTS FROM DASHBOARD
# =====================================================

if len(sys.argv) < 3:
    print("Usage: python script.py [dataset_path] [num_clients]")
    sys.exit(1)

dataset_path = sys.argv[1]
num_clients = int(sys.argv[2])

print(f"[SPLIT 235] Dataset: {dataset_path}")
print(f"[SPLIT 235] Num clients: {num_clients}")

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv(dataset_path, sep=None, engine="python")

# Shuffle strict identique (mix global 42 sur le bloc 235)
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# =====================================================
# OUTPUT FOLDER
# =====================================================

output_dir = "generated_splits_235"
os.makedirs(output_dir, exist_ok=True)

# =====================================================
# GENERATE SPLITS WITH EXACT INDICES
# =====================================================

client_parts = []

if num_clients == 2:
    client_parts.append(df_shuffled.iloc[:94])
    client_parts.append(df_shuffled.iloc[94:])

elif num_clients == 3:
    client_parts.append(df_shuffled.iloc[:71])
    client_parts.append(df_shuffled.iloc[71:177])
    client_parts.append(df_shuffled.iloc[177:])

elif num_clients == 4:
    client_parts.append(df_shuffled.iloc[:71])
    client_parts.append(df_shuffled.iloc[71:118])
    client_parts.append(df_shuffled.iloc[118:153])
    client_parts.append(df_shuffled.iloc[153:])

elif num_clients == 5:
    # Utilisation forcée de df_shuffled pour corriger la coquille du script "avant"
    client_parts.append(df_shuffled.iloc[:47])
    client_parts.append(df_shuffled.iloc[47:118])
    client_parts.append(df_shuffled.iloc[118:153])
    client_parts.append(df_shuffled.iloc[153:212])
    client_parts.append(df_shuffled.iloc[212:])

else:
    raise ValueError("Unsupported number of clients")

# =====================================================
# SAVE FILES
# =====================================================

files = []
for i, part in enumerate(client_parts):
    file_path = os.path.join(output_dir, f"client{i+1}.csv")
    
    # NOTE: index=False comme avant. Si ton modèle plante, remets sep="," 
    part.to_csv(file_path, index=False, sep=";")
    files.append(file_path)
    print(f"[SPLIT 235] Client {i+1} saved -> {file_path} ({len(part)} rows)")

print("\n[SPLIT 235 DONE]")
print(files)