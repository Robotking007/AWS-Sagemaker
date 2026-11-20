"""Define, execute, and deploy the SageMaker training pipeline."""

import argparse
import json
from pathlib import Path

import boto3
import pandas as pd
from sagemaker.deserializers import JSONDeserializer
from sagemaker.inputs import TrainingInput
from sagemaker.serializers import CSVSerializer
from sagemaker.session import Session
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel, SKLearnPredictor
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import TrainingStep

ROOT = Path(__file__).resolve().parent
FRAMEWORK = "1.2-1"
METRICS = ("accuracy", "precision", "recall", "f1", "auc")


def build_pipeline(args, session):
    estimator = SKLearn(
        entry_point=str(ROOT / "train.py"),
        role=args.role,
        framework_version=FRAMEWORK,
        py_version="py3",
        instance_count=1,
        instance_type="ml.m5.large",
        max_run=900,
        code_location=f"s3://{args.bucket}/{args.prefix}/code",
        output_path=f"s3://{args.bucket}/{args.prefix}/output",
        hyperparameters={"n-estimators": 200, "max-depth": 8},
        metric_definitions=[
            {"Name": f"test:{name}", "Regex": rf"test_{name}=([0-9.]+)"}
            for name in METRICS
        ],
        sagemaker_session=session,
    )
    channels = ("train", "test")
    parameters = {
        name: ParameterString(
            name=f"{name.title()}Data",
            default_value=f"s3://{args.bucket}/{args.prefix}/input/{name}/{name}.csv",
        )
        for name in channels
    }
    step = TrainingStep(
        name="TrainEvaluateRandomForest",
        step_args=estimator.fit(
            {name: TrainingInput(uri, content_type="text/csv") for name, uri in parameters.items()}
        ),
    )
    return Pipeline(
        name=args.pipeline_name,
        parameters=list(parameters.values()),
        steps=[step],
        sagemaker_session=session,
    )


def submit_training(args, aws):
    session = PipelineSession(boto_session=aws, default_bucket=args.bucket)
    s3 = aws.client("s3")
    for name in ("train", "test"):
        s3.upload_file(
            str(ROOT / "data" / f"{name}.csv"),
            args.bucket,
            f"{args.prefix}/input/{name}/{name}.csv",
        )
    pipeline = build_pipeline(args, session)
    pipeline.upsert(role_arn=args.role)
    execution = pipeline.start()
    print(f"Pipeline execution: {execution.arn}", flush=True)
    execution.wait(delay=15, max_attempts=80)
    step = execution.list_steps()[0]
    job_name = step["Metadata"]["TrainingJob"]["Arn"].rsplit("/", 1)[-1]
    job = aws.client("sagemaker").describe_training_job(TrainingJobName=job_name)
    print(json.dumps({
        "training_job": job_name,
        "model_data": job["ModelArtifacts"]["S3ModelArtifacts"],
        "training_seconds": job.get("TrainingTimeInSeconds"),
        "billable_seconds": job.get("BillableTimeInSeconds"),
        "metrics": {item["MetricName"]: item["Value"] for item in job.get("FinalMetricDataList", [])},
    }, indent=2))


def deploy(args, aws):
    model = SKLearnModel(
        name=args.endpoint,
        model_data=args.model_data,
        role=args.role,
        entry_point=str(ROOT / "train.py"),
        framework_version=FRAMEWORK,
        py_version="py3",
        sagemaker_session=Session(boto_session=aws, default_bucket=args.bucket),
    )
    model.deploy(
        initial_instance_count=1,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint,
    )
    print(f"Endpoint {args.endpoint} is InService")


def predict(args, aws):
    predictor = SKLearnPredictor(
        endpoint_name=args.endpoint,
        serializer=CSVSerializer(),
        deserializer=JSONDeserializer(),
        sagemaker_session=Session(boto_session=aws),
    )
    test = pd.read_csv(ROOT / "data" / "test.csv").head(5)
    print("Predictions:", predictor.predict(test.drop(columns="target").to_numpy()))
    print("Actual:     ", test["target"].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="ap-south-1")
    parser.add_argument("--endpoint", default="breast-cancer-rf-endpoint")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--pipeline-name", default="BreastCancerRandomForest")
    training.add_argument("--prefix", default="breast-cancer")
    deployment = commands.add_parser("deploy")
    deployment.add_argument("--model-data", required=True)
    deployment.add_argument("--instance-type", default="ml.m5.large")
    for command in (training, deployment):
        command.add_argument("--bucket", required=True)
        command.add_argument("--role", required=True)
    commands.add_parser("predict")
    args = parser.parse_args()
    aws = boto3.Session(region_name=args.region)
    actions = {"train": submit_training, "deploy": deploy, "predict": predict}
    actions[args.command](args, aws)


if __name__ == "__main__":
    main()
