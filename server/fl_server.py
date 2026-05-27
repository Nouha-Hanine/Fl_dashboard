import flwr as fl
import joblib
import pickle
import pandas as pd
import warnings
import os
import json

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning
)


###################################################
# LECTURE CONFIG DASHBOARD
###################################################

with open("server/config.json","r") as f:
    CONFIG=json.load(f)

MODEL_PATH=CONFIG["model"]
K_VALUE=CONFIG["k"]
NB_CLIENTS=CONFIG["clients"]


print("\n========================")
print("FL CONFIG")
print("========================")
print("Model :",MODEL_PATH)
print("K :",K_VALUE)
print("Clients :",NB_CLIENTS)
print("========================")


###################################################
# LOAD MODEL
###################################################

try:

    GM=joblib.load(
        MODEL_PATH
    )

    print(
        f"\n[SERVER] Initial model loaded "
        f"({len(GM.estimators_)} trees)"
    )

except:

    raise FileNotFoundError(
        "Initial model not found"
    )


###################################################
# LOAD GLOBAL TEST
###################################################

test_data=pd.read_csv(
    "data/test_Data.csv",
    sep=";"
)

target="Progression_Status"

X_test_raw=test_data.drop(
    columns=[target]
)

y_test=test_data[target]

X_test_raw=pd.get_dummies(
    X_test_raw
)


print("\n======================")
print("TEST CLASS DISTRIBUTION")
print("======================")
print(
    y_test.value_counts()
)
print("======================\n")


###################################################
# ALIGN FEATURES
###################################################

def align_features(
        X,
        model
):

    expected_cols=(
        model.feature_names_in_
    )

    X_aligned=X.copy()

    for col in expected_cols:

        if col not in X_aligned.columns:

            X_aligned[col]=0


    X_aligned=(
        X_aligned[expected_cols]
    )

    return X_aligned


###################################################
# SAVE DASHBOARD METRICS
###################################################

def save_dashboard_metrics(
        round_num,
        acc,
        sensitivity,
        specificity,
        precision,
        f1,
        auc
):

    metrics={

        "round":round_num,

        "accuracy":
        float(acc),

        "sensitivity":
        float(sensitivity),

        "specificity":
        float(specificity),

        "precision":
        float(precision),

        "f1":
        float(f1),

        "auc":
        float(auc)

    }

    with open(
        "dashboard/global_metrics.json",
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )


###################################################
# FED STRATEGY
###################################################

class FedTreeStrategy(
        fl.server.strategy.FedAvg
):

    def __init__(
            self,
            *args,
            **kwargs
    ):

        super().__init__(
            *args,
            **kwargs
        )

        self.stop_after_evaluate=False


###################################################

    def initialize_parameters(
            self,
            client_manager
    ):

        print(
            "[SERVER] Sending initial model..."
        )

        config={

            "model":GM,

            "K":K_VALUE,

            "params":
            GM.get_params()

        }

        return fl.common.ndarrays_to_parameters(

            [pickle.dumps(config)]

        )


###################################################

    def aggregate_fit(
            self,
            server_round,
            results,
            failures
    ):

        global GM

        new_trees=[]

        print(
            f"\n===== ROUND "
            f"{server_round}"
            f" ====="
        )

        for i,(_,fit_res) in enumerate(results):

            bij=fit_res.metrics.get(
                "bij"
            )

            tij_bytes=fit_res.metrics.get(
                "tij"
            )


            if bij:

                print(
                    f"Client {i+1}: bij=True"
                )

            elif(
                    not bij
                    and tij_bytes
            ):

                trees=pickle.loads(
                    tij_bytes
                )

                new_trees.extend(
                    trees
                )

                print(

                    f"Client {i+1}: "

                    f"{len(trees)} "

                    f"trees received"

                )

            else:

                print(
                    f"Client{i+1}"
                    f":invalid"
                )


        if len(new_trees)>0:

            GM.estimators_.extend(
                new_trees
            )

            GM.n_estimators=(
                len(
                    GM.estimators_
                )
            )

            print(
                f"[SERVER]"
                f"{len(new_trees)}"
                f"trees added"
            )

            print(
                f"[SERVER]"
                f"New model size:"
                f"{GM.n_estimators}"
            )

        else:

            print(
                "[EARLY STOPPING]"
            )

            self.stop_after_evaluate=True


        config_next={

            "model":GM,

            "K":K_VALUE,

            "params":
            GM.get_params()

        }


        return(

            fl.common.ndarrays_to_parameters(

                [pickle.dumps(
                    config_next
                )]

            ),

            {}

        )


###################################################

    def evaluate_global_model(
            self,
            server_round
    ):

        global GM


        print("\n")
        print("="*50)

        print(
            f"GLOBAL MODEL "
            f"ROUND "
            f"{server_round}"
        )

        print("="*50)


        X_test=align_features(
            X_test_raw,
            GM
        )


        y_pred=GM.predict(
            X_test
        )


        y_prob=(
            GM.predict_proba(
                X_test
            )[:,1]
        )


        acc=accuracy_score(
            y_test,
            y_pred
        )


        sensitivity=recall_score(
            y_test,
            y_pred,
            zero_division=0
        )


        tn,fp,fn,tp=confusion_matrix(
            y_test,
            y_pred
        ).ravel()


        specificity=(
            tn/(tn+fp)
        )


        precision=precision_score(
            y_test,
            y_pred
        )


        f1=f1_score(
            y_test,
            y_pred
        )


        auc=roc_auc_score(
            y_test,
            y_prob
        )


        print(
            f"Accuracy:{acc:.4f}"
        )

        print(
            f"Sensitivity:{sensitivity:.4f}"
        )

        print(
            f"Specificity:{specificity:.4f}"
        )

        print(
            f"Precision:{precision:.4f}"
        )

        print(
            f"F1:{f1:.4f}"
        )

        print(
            f"AUC:{auc:.4f}"
        )


        save_dashboard_metrics(

            server_round,

            acc,

            sensitivity,

            specificity,

            precision,

            f1,

            auc

        )


###################################################

    def aggregate_evaluate(
            self,
            server_round,
            results,
            failures
    ):

        loss,_=super().aggregate_evaluate(

            server_round,

            results,

            failures

        )

        self.evaluate_global_model(
            server_round
        )


        if(
            self.stop_after_evaluate
        ):

            print(
                "\nFINAL STOP"
            )

            joblib.dump(

                GM,

                "FINAL_federated_model.pkl"

            )

            raise StopIteration(
                "Done"
            )


        return loss,{}


###################################################

if __name__=="__main__":


    strategy=FedTreeStrategy(

        min_fit_clients=
        NB_CLIENTS,

        min_available_clients=
        NB_CLIENTS,

        min_evaluate_clients=
        NB_CLIENTS

    )


    try:

        fl.server.start_server(

            server_address=
            "0.0.0.0:8080",

            config=
            fl.server.ServerConfig(

                num_rounds=15

            ),

            strategy=
            strategy

        )


    except StopIteration as e:

        print(e)


    print("\n")

    print("="*50)

    print(
        "TRAINING FINISHED"
    )

    print("="*50)


    os._exit(0)