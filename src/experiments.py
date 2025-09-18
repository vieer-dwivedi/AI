import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import subprocess
from itertools import product

# ------------------------------
# Load processed data
train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")
test = pd.read_csv("data/processed/test.csv")

X_train, y_train = train.drop("y", axis=1), train["y"]
X_val, y_val = val.drop("y", axis=1), val["y"]
X_test, y_test = test.drop("y", axis=1), test["y"]

# ------------------------------
# Get current data version (Git commit hash)
data_version = subprocess.getoutput("git rev-parse HEAD")
print("Using data version:", data_version)

# ------------------------------
# Set remote MLflow server
mlflow.set_tracking_uri("http://54.172.80.91:8080")
mlflow.set_experiment("bank_marketing_remote")

# ------------------------------
# Define hyperparameter grid
n_estimators_list = [50, 100, 200]
max_depth_list = [5, 10, None]

# ------------------------------
# Run experiments
for n_estimators, max_depth in product(n_estimators_list, max_depth_list):
    with mlflow.start_run():
        # Log hyperparameters and data version
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("data_version", data_version)
        
        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        val_prec = precision_score(y_val, y_val_pred)
        val_rec = recall_score(y_val, y_val_pred)
        
        # Log metrics
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("val_f1", val_f1)
        mlflow.log_metric("val_precision", val_prec)
        mlflow.log_metric("val_recall", val_rec)
        
        # Evaluate on test set
        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred)
        test_rec = recall_score(y_test, y_test_pred)
        
        # Log test metrics
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("test_precision", test_prec)
        mlflow.log_metric("test_recall", test_rec)
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"✅ Run logged: n_estimators={n_estimators}, max_depth={max_depth}, val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")
