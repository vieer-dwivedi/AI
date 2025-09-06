import boto3, os

sm = boto3.client("sagemaker")

MODEL_GROUP = "my-model-group"
ENDPOINT = "my-prod-endpoint"

# 1. Get latest Approved model package
resp = sm.list_model_packages(
    ModelPackageGroupName=MODEL_GROUP,
    ModelApprovalStatus="Approved",
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=1
)
model_package_arn = resp["ModelPackageSummaryList"][0]["ModelPackageArn"]

# 2. Create SageMaker Model
model_name = f"{MODEL_GROUP}-model"
sm.create_model(
    ModelName=model_name,
    PrimaryContainer={
        "ModelPackageName": model_package_arn
    },
    ExecutionRoleArn=os.environ["SAGEMAKER_ROLE_ARN"]
)

# 3. Update Endpoint
try:
    sm.describe_endpoint(EndpointName=ENDPOINT)
    sm.update_endpoint(
        EndpointName=ENDPOINT,
        EndpointConfigName=model_name
    )
    print("Updated endpoint")
except sm.exceptions.ClientError:
    sm.create_endpoint(
        EndpointName=ENDPOINT,
        EndpointConfigName=model_name
    )
    print("Created endpoint")
