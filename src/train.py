import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib

# -------------------------
# Configuration
MLFLOW_URI = "http://54.172.80.91:8080"
EXPERIMENT_NAME = "bank_marketing_remote"
REGISTERED_MODEL_NAME = "BankMarketing_RF"
LOCAL_MODEL_PATH = "models/best_model.pkl"  # your local model path
ARTIFACT_PATH = "best_model"  # path inside MLflow run

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()

# -------------------------
# 1. Get best run by test_f1
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
experiment_id = experiment.experiment_id

runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.test_f1 DESC"]
)

best_run = runs.iloc[0]
best_run_id = best_run.run_id
best_f1 = best_run["metrics.test_f1"]
print(f"Best run ID: {best_run_id}, test_f1: {best_f1}")

# -------------------------
# 2. Always log the local model in a new nested run
model = joblib.load(LOCAL_MODEL_PATH)
with mlflow.start_run(run_name="register_local_best_model", nested=True):
    mlflow.sklearn.log_model(model, artifact_path=ARTIFACT_PATH)
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/{ARTIFACT_PATH}"

# -------------------------
# 3. Register the model
registered_model = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
print(f"Successfully registered model '{REGISTERED_MODEL_NAME}'")

# -------------------------
# 4. Transition to Production
version = registered_model.version
client.transition_model_version_stage(
    name=REGISTERED_MODEL_NAME,
    version=version,
    stage="Production"
)
print(f"Model version {version} is now in Production")
