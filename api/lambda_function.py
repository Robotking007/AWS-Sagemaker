"""Validate an API Gateway request and invoke SageMaker."""

import json
import logging
import math
import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError

LOGGER = logging.getLogger(__name__)
RUNTIME = boto3.client("sagemaker-runtime")


def response(status, payload):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }


def lambda_handler(event, context):
    try:
        features = json.loads(event.get("body") or "{}")["features"]
        if not isinstance(features, list) or len(features) != 30:
            raise ValueError("Wrong feature count")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in features):
            raise ValueError("Invalid feature value")
    except (KeyError, TypeError, ValueError, OverflowError):
        return response(400, {"error": "Provide exactly 30 finite numeric features"})

    try:
        result = RUNTIME.invoke_endpoint(
            EndpointName=os.environ["ENDPOINT_NAME"],
            ContentType="text/csv",
            Accept="application/json",
            Body=",".join(str(value) for value in features).encode(),
        )
        label = json.loads(result["Body"].read())[0]
        prediction = {0: "malignant", 1: "benign"}[label]
    except (BotoCoreError, ClientError, TypeError, ValueError, IndexError, KeyError):
        LOGGER.exception("SageMaker invocation failed")
        return response(502, {"error": "Prediction service unavailable"})
    return response(200, {"label": label, "prediction": prediction})
