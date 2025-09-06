# pipeline/pipeline.py
import os, time, sagemaker
from sagemaker.workflow.parameters import ParameterString, ParameterFloat, ParameterInteger
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.functions import Join

sess = sagemaker.session.Session()
region = sess.boto_region_name
role = sagemaker.get_execution_role()

# Parameters
input_s3 = ParameterString("InputDataS3Uri")
model_pkg_group = ParameterString("ModelPackageGroupName")
acc_threshold = ParameterFloat("AccuracyThreshold", default_value=0.90)
instance_type = ParameterString("TrainingInstanceType", default_value="ml.m5.xlarge")
instance_count = ParameterInteger("TrainingInstanceCount", default_value=1)

# Processing step (feature engineering)
processor = Processor(
    image_uri=sagemaker.image_uris.retrieve("sklearn", region),
    role=role, instance_count=1, instance_type="ml.m5.large"
)
step_process = ProcessingStep(
    name="Process",
    processor=processor,
    inputs=[ProcessingInput(source=input_s3, destination="/opt/ml/processing/input")],
    outputs=[ProcessingOutput(source="/opt/ml/processing/train", destination=Join(on="/", values=[sess.default_bucket(), "processed/train"])),
             ProcessingOutput(source="/opt/ml/processing/test", destination=Join(on="/", values=[sess.default_bucket(), "processed/test"]))],
    code="src/processing.py",
)

# Training step
estimator = Estimator(
    image_uri=sagemaker.image_uris.retrieve("xgboost", region, version="1.7-1"),
    role=role, instance_type=instance_type, instance_count=instance_count,
    output_path=f"s3://{sess.default_bucket()}/model-artifacts/"
)
step_train = TrainingStep(
    name="Train",
    estimator=estimator,
    inputs={"train": step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            "validation": step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri},
)

# Evaluation step
eval_processor = processor
step_eval = ProcessingStep(
    name="Evaluate",
    processor=eval_processor,
    inputs=[
        ProcessingInput(source=step_train.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model"),
        ProcessingInput(source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri, destination="/opt/ml/processing/test")
    ],
    outputs=[ProcessingOutput(source="/opt/ml/processing/evaluation", destination=f"s3://{sess.default_bucket()}/evaluation")],
    code="src/evaluate.py",  # writes evaluation.json with {"accuracy": float}
)

model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=Join(on="/", values=[step_eval.outputs[0].destination, "evaluation.json"]),
        content_type="application/json",
    )
)

# Register to Model Registry (Pending manual approval)
step_register = RegisterModel(
    name="RegisterModel",
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"], response_types=["text/csv"],
    inference_instances=["ml.m5.large"], transform_instances=["ml.m5.large"],
    model_package_group_name=model_pkg_group,
    model_metrics=model_metrics,
    approval_status="PendingManualApproval",
)

# Gate on metric threshold
cond = ConditionLessThanOrEqualTo(
    left_json_path="accuracy",
    right=acc_threshold,
    operand="values"
)
step_condition = ConditionStep(
    name="CheckMetrics",
    conditions=[cond],
    if_steps=[step_register],
    else_steps=[],
)

pipeline = Pipeline(
    name=f"demo-{int(time.time())}",
    parameters=[input_s3, model_pkg_group, acc_threshold, instance_type, instance_count],
    steps=[step_process, step_train, step_eval, step_condition],
    sagemaker_session=sess,
)
