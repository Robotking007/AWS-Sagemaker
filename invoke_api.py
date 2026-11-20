"""Send a signed sample request to the lab's API Gateway endpoint."""

import argparse
import csv
import json
from pathlib import Path
from urllib.request import Request, urlopen

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--region", default="ap-south-1")
    args = parser.parse_args()

    with (Path(__file__).resolve().parent / "data" / "test.csv").open(newline="") as dataset:
        row = next(csv.DictReader(dataset))
    actual = int(row.pop("target"))
    payload = json.dumps({"features": [float(value) for value in row.values()]}).encode()
    credentials = boto3.Session().get_credentials()
    if credentials is None:
        parser.error("Configure AWS credentials before invoking an IAM-authenticated API")

    signed = AWSRequest(
        method="POST", url=args.url, data=payload, headers={"Content-Type": "application/json"}
    )
    SigV4Auth(credentials.get_frozen_credentials(), "execute-api", args.region).add_auth(signed)
    request = Request(args.url, data=payload, headers=dict(signed.headers), method="POST")
    with urlopen(request, timeout=30) as result:
        print(result.read().decode())
    print(f"Actual label: {actual}")


if __name__ == "__main__":
    main()
