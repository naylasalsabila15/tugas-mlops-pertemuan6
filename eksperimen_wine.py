import hashlib
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
from mlflow import MlflowClient

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_predict, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. SETUP MLFLOW & DATA
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("wine-quality-classification")

data = load_wine(as_frame=True)
df = data.frame
data_md5 = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["target"], random_state=42)
X_train, y_train = train_df.drop(columns="target"), train_df["target"]
X_test, y_test = test_df.drop(columns="target"), test_df["target"]

train_ds = mlflow.data.from_pandas(train_df, name="wine-train", targets="target")
COMMON_TAGS = {"data_md5": data_md5, "split_seed": "42", "dataset": "sklearn-wine"}

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = ["f1_macro", "precision_macro", "recall_macro"]

# 2. FUNGSI LOGGING SENTRAL
def run_experiment(run_name, pipe, tags=None, note=""):
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({**COMMON_TAGS, "candidate": "true", **(tags or {})})
        mlflow.set_tag("mlflow.note.content", note)
        mlflow.log_input(train_ds, context="training")
        
        mlflow.log_param("pipeline_steps", " -> ".join(pipe.named_steps))
        mlflow.log_params({f"clf_{k}": v for k, v in pipe[-1].get_params().items()})
        
        cv = cross_validate(pipe, X_train, y_train, cv=CV, scoring=SCORING)
        for m in SCORING:
            mlflow.log_metric(f"cv_{m}_mean", cv[f"test_{m}"].mean())
            mlflow.log_metric(f"cv_{m}_std", cv[f"test_{m}"].std())
            
        y_oof = cross_val_predict(pipe, X_train, y_train, cv=CV)
        disp = ConfusionMatrixDisplay.from_predictions(y_train, y_oof)
        mlflow.log_figure(disp.figure_, "plots/confusion_matrix_oof.png")
        plt.close(disp.figure_)
        
        pipe.fit(X_train, y_train)
        signature = infer_signature(X_train, pipe.predict(X_train))
        
        # --- PERBAIKAN: Menambahkan skops_trusted_types ---
        info = mlflow.sklearn.log_model(
            pipe, 
            name="model", 
            signature=signature, 
            input_example=X_train.head(3),
            skops_trusted_types=["sklearn.neural_network._stochastic_optimizers.AdamOptimizer", "sklearn.tree._tree.Tree"]
        )
        
        mlflow.set_tag("model_uri", info.model_uri)
        return run.info.run_id

if __name__ == "__main__":
    print("Memulai eksperimen baseline & algoritma sederhana...")
    
    # 3. EKSPERIMEN SEDERHANA
    run_experiment("00-dummy-baseline", 
                   Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
                   tags={"stage": "baseline", "model_family": "dummy"}, note="Batas bawah performa.")
    
    run_experiment("01-gnb-default", 
                   Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())]),
                   tags={"stage": "model-selection", "model_family": "gnb"}, note="GNB dengan standardisasi.")
                   
    run_experiment("02-mlp-default", 
                   Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(max_iter=1000, random_state=42))]),
                   tags={"stage": "model-selection", "model_family": "mlp"}, note="Multi-layer Perceptron (NN).")

    print("Memulai hyperparameter tuning Random Forest (Nested Runs)...")
    
    # 4. TUNING DENGAN NESTED RUNS
    pipe_rf = Pipeline([("clf", RandomForestClassifier(random_state=42))])
    grid = {"clf__n_estimators": [50, 100], "clf__max_depth": [3, 5, None]}
    gs = GridSearchCV(pipe_rf, grid, cv=CV, scoring="f1_macro", n_jobs=-1)

    with mlflow.start_run(run_name="03-rf-gridsearch", tags={**COMMON_TAGS, "stage": "tuning"}):
        mlflow.log_input(train_ds, context="training")
        gs.fit(X_train, y_train)
        res = gs.cv_results_
        
        for i, params in enumerate(res["params"]):
            with mlflow.start_run(run_name=f"rf-trial-{i:02d}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("cv_f1_macro_mean", res["mean_test_score"][i])
                mlflow.log_metric("cv_f1_macro_std", res["std_test_score"][i])
                
        mlflow.log_params(gs.best_params_)
        mlflow.log_metric("best_cv_f1_macro", gs.best_score_)
        
        signature = infer_signature(X_train, gs.best_estimator_.predict(X_train))
        
        # --- PERBAIKAN: Menambahkan skops_trusted_types untuk Random Forest ---
        info = mlflow.sklearn.log_model(
            gs.best_estimator_, 
            name="model", 
            signature=signature,
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        
        mlflow.set_tag("model_uri", info.model_uri)
        best_model_uri = info.model_uri

    print("Evaluasi model terbaik pada Data Test dan Registrasi ke Registry...")
    
    # 5. EVALUASI FINAL & REGISTRY
    model = mlflow.sklearn.load_model(best_model_uri)
    y_pred = model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    
    print(f"Final Test F1-Macro: {test_f1:.4f}")

    client = MlflowClient()
    model_name = "Wine-Quality-Classifier"
    mv = mlflow.register_model(best_model_uri, model_name)
    client.update_model_version(model_name, mv.version, description=f"Model RF Tuned. data_md5={data_md5}")

    if test_f1 >= 0.90:
        client.set_registered_model_alias(model_name, "champion", mv.version)
        print("Model dipromosikan sebagai @champion!")
    else:
        client.set_registered_model_alias(model_name, "challenger", mv.version)
        print("Performa kurang, masuk status @challenger.")
        
    print("Eksperimen selesai! Silakan cek http://127.0.0.1:5000")