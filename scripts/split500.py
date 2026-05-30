import pandas as pd


def split_500(df, num_clients):
    """
    Dynamic federated learning split for dataset 500.
    Generates client datasets based on selected number of clients.
    """

    # 1. Shuffle dataset (reproducible)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    output_dir = "orchestrator/client_data"

    print(f"\n[500 SPLIT] Splitting into {num_clients} clients...")

    # =====================================================
    # 2 CLIENTS
    # =====================================================
    if num_clients == 2:

        split1 = df.iloc[:200]
        split2 = df.iloc[200:]

        split1.to_csv(f"{output_dir}/client1.csv", index=False)
        split2.to_csv(f"{output_dir}/client2.csv", index=False)

    # =====================================================
    # 3 CLIENTS
    # =====================================================
    elif num_clients == 3:

        split1 = df.iloc[:150]
        split2 = df.iloc[150:375]
        split3 = df.iloc[375:]

        split1.to_csv(f"{output_dir}/client1.csv", index=False)
        split2.to_csv(f"{output_dir}/client2.csv", index=False)
        split3.to_csv(f"{output_dir}/client3.csv", index=False)

    # =====================================================
    # 4 CLIENTS
    # =====================================================
    elif num_clients == 4:

        split1 = df.iloc[:150]
        split2 = df.iloc[150:250]
        split3 = df.iloc[250:325]
        split4 = df.iloc[325:]

        split1.to_csv(f"{output_dir}/client1.csv", index=False)
        split2.to_csv(f"{output_dir}/client2.csv", index=False)
        split3.to_csv(f"{output_dir}/client3.csv", index=False)
        split4.to_csv(f"{output_dir}/client4.csv", index=False)

    # =====================================================
    # 5 CLIENTS
    # =====================================================
    elif num_clients == 5:

        split1 = df.iloc[:100]
        split2 = df.iloc[100:250]
        split3 = df.iloc[250:325]
        split4 = df.iloc[325:450]
        split5 = df.iloc[450:]

        split1.to_csv(f"{output_dir}/client1.csv", index=False)
        split2.to_csv(f"{output_dir}/client2.csv", index=False)
        split3.to_csv(f"{output_dir}/client3.csv", index=False)
        split4.to_csv(f"{output_dir}/client4.csv", index=False)
        split5.to_csv(f"{output_dir}/client5.csv", index=False)

    else:
        raise ValueError("num_clients must be 2, 3, 4, or 5")

    print("[500 SPLIT] Done ✔")