import optuna
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import subprocess
import joblib
# ------------------------------
# Load processed data
train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")
test = pd.read_csv("data/processed/test.csv")

X_train, y_train = train.drop("y", axis=1), train["y"]
X_val, y_val = val.drop("y", axis=1), val["y"]
X_test, y_test = test.drop("y", axis=1), test["y"]

# ------------------------------
# Remote MLflow server
mlflow.set_tracking_uri("http://54.172.80.91:8080")

# Use existing experiment (baseline runs exist here)
baseline_experiment_name = "bank_marketing_remote"
mlflow.set_experiment(baseline_experiment_name)

# ------------------------------
# Get data version
data_version = subprocess.getoutput("git rev-parse HEAD")
print(f"Using data version: {data_version}")

# ------------------------------
# Optuna objective function
def objective(trial):
    # Hyperparameter search space
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 15)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Validation predictions
    y_val_pred = model.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred)

    # Log run to MLflow (nested run)
    with mlflow.start_run(nested=True):
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("data_version", data_version)

        # Validation metrics
        mlflow.log_metric("val_f1", val_f1)

        # Test metrics
        y_test_pred = model.predict(X_test)
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_test_pred))
        mlflow.log_metric("test_f1", f1_score(y_test, y_test_pred))
        mlflow.log_metric("test_precision", precision_score(y_test, y_test_pred))
        mlflow.log_metric("test_recall", recall_score(y_test, y_test_pred))

        # Log model
        mlflow.sklearn.log_model(model, "model")

    return val_f1

# ------------------------------
# Run Optuna hyperparameter tuning
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)  # increase n_trials for more exhaustive search

# ------------------------------
# Print best hyperparameters
print("✅ Hyperparameter tuning complete!")
print("Best hyperparameters found:", study.best_params)
print(f"Best validation F1: {study.best_value:.4f}")

# ------------------------------
# Retrain best model on train + validation and log
best_params = study.best_params
best_model = RandomForestClassifier(**best_params, random_state=42)
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])
best_model.fit(X_train_val, y_train_val)

with mlflow.start_run(nested=True):
    mlflow.log_param("data_version", data_version)
    for k, v in best_params.items():
        mlflow.log_param(k, v)

    y_test_pred = best_model.predict(X_test)
    mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_test_pred))
    mlflow.log_metric("test_f1", f1_score(y_test, y_test_pred))
    mlflow.log_metric("test_precision", precision_score(y_test, y_test_pred))
    mlflow.log_metric("test_recall", recall_score(y_test, y_test_pred))

    mlflow.sklearn.log_model(best_model, "best_model")
    print("✅ Best model retrained on train+val and logged to MLflow.")
    
    joblib.dump(best_model, "models/best_model.pkl")

