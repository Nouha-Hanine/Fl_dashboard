import pandas as pd
import sys
import os

# =====================================================
# INPUTS FROM DASHBOARD
# =====================================================

# dataset path envoyé par FastAPI / dashboard
dataset_path = sys.argv[1]

# nombre de clients choisi dans le dashboard
num_clients = int(sys.argv[2])

print(f"[SPLIT 235] Dataset: {dataset_path}")
print(f"[SPLIT 235] Num clients: {num_clients}")

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv(dataset_path, sep=None, engine="python")

# shuffle dataset
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# =====================================================
# OUTPUT FOLDER
# =====================================================

output_dir = "generated_splits_235"
os.makedirs(output_dir, exist_ok=True)

# =====================================================
# DEFINE SPLITS (TA LOGIQUE ORIGINALE)
# =====================================================

if num_clients == 2:
    ratios = [0.40, 0.60]

elif num_clients == 3:
    ratios = [0.30, 0.45, 0.25]

elif num_clients == 4:
    ratios = [0.30, 0.20, 0.15, 0.35]

elif num_clients == 5:
    ratios = [0.20, 0.30, 0.15, 0.25, 0.10]

else:
    raise ValueError("Unsupported number of clients")

# =====================================================
# GENERATE SPLITS
# =====================================================

start = 0
files = []

n = len(df_shuffled)

for i, ratio in enumerate(ratios):

    # last split takes remaining rows (avoid rounding issues)
    if i == len(ratios) - 1:
        part = df_shuffled.iloc[start:]
    else:
        end = start + int(n * ratio)
        part = df_shuffled.iloc[start:end]

    file_path = os.path.join(output_dir, f"client{i+1}.csv")

    part.to_csv(file_path, index=False)

    files.append(file_path)

    print(f"[SPLIT 235] Client {i+1} saved -> {file_path} ({len(part)} rows)")

    start = end

# =====================================================
# RESULT
# =====================================================

print("\n[SPLIT 235 DONE]")
print(files)