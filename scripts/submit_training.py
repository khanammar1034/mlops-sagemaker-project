import os
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

ROLE = os.environ.get(
    "SAGEMAKER_ROLE",
    "arn:aws:iam::660864587818:role/SageMakerExecutionRole"
)

session = sagemaker.Session()

estimator = SKLearn(
    entry_point="train.py",
    source_dir="src",
    role=ROLE,
    framework_version="1.2-1",
    py_version="py3",
    instance_count=1,
    instance_type="ml.m5.large",
    sagemaker_session=session,
)

print("Submitting training job...")

estimator.fit(wait=True)

print("Training job submitted successfully.")