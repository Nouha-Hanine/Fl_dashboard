import pandas as pd


def split_368(df, num_clients):
    """
    Split dataset 368 dynamically based on number of FL clients
    and generate client files for federated learning.
    """

    # 1. Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    output_dir = "orchestrator/client_data"

    print(f"\n[368 SPLIT] Splitting into {num_clients} clients...")

    # =====================================================
    # 2 CLIENTS
    # =====================================================
    if num_clients == 2:

        split1 = df.iloc[:147]
        split2 = df.iloc[147:]

        split1.to_csv(f"{output_dir}/client1.csv", index=False)
        split2.to_csv(f"{output_dir}/client2.csv", index=False)

    # =====================================================
    # 3 CLIENTS
    # =====================================================
    elif num_clients == 3:

        split1 = df.iloc[:110]
        split2 = df.iloc[110:276]
        split3 = df.iloc[276:]

        split1.to_csv(f"{output_dir}/client1.csv", index=False)
        split2.to_csv(f"{output_dir}/client2.csv", index=False)
        split3.to_csv(f"{output_dir}/client3.csv", index=False)

    # =====================================================
    # 4 CLIENTS
    # =====================================================
    elif num_clients == 4:

        split1 = df.iloc[:110]
        split2 = df.iloc[110:184]
        split3 = df.iloc[184:239]
        split4 = df.iloc[239:]

        split1.to_csv(f"{output_dir}/client1.csv", index=False)
        split2.to_csv(f"{output_dir}/client2.csv", index=False)
        split3.to_csv(f"{output_dir}/client3.csv", index=False)
        split4.to_csv(f"{output_dir}/client4.csv", index=False)

    # =====================================================
    # 5 CLIENTS
    # =====================================================
    elif num_clients == 5:

        split1 = df.iloc[:74]
        split2 = df.iloc[74:184]
        split3 = df.iloc[184:239]
        split4 = df.iloc[239:331]
        split5 = df.iloc[331:]

        split1.to_csv(f"{output_dir}/client1.csv", index=False)
        split2.to_csv(f"{output_dir}/client2.csv", index=False)
        split3.to_csv(f"{output_dir}/client3.csv", index=False)
        split4.to_csv(f"{output_dir}/client4.csv", index=False)
        split5.to_csv(f"{output_dir}/client5.csv", index=False)

    else:
        raise ValueError("num_clients must be 2, 3, 4, or 5")

    print(f"[368 SPLIT] Done ✔")