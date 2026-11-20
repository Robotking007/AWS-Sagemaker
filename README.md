# Assignment 9 — AWS SageMaker ML Pipeline

Train and evaluate a scikit-learn Random Forest on the Wisconsin Diagnostic Breast Cancer dataset, then serve predictions from a SageMaker real-time endpoint behind an IAM-authenticated API Gateway REST API. The dataset contains 455 training rows and 114 test rows, with 30 numeric features. The target uses `0` for malignant and `1` for benign.

## Local run

Use Python 3.10. The scikit-learn version matches the SageMaker training image. Open `assignment9.ipynb` for the notebook workflow, or run these PowerShell commands from the repository root.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python train.py --n-estimators 100 --max-depth 4 --model-dir artifacts/rf100
.venv\Scripts\python train.py --n-estimators 200 --max-depth 8 --model-dir artifacts/rf200
```

## Managed training and inference

Configure AWS credentials, the AWS CLI, and the AWS SAM CLI. Use a private, encrypted S3 bucket in the selected region and a SageMaker execution role trusted by `sagemaker.amazonaws.com`, with bucket access scoped to the lab objects. The caller needs permission to pass that role and manage the pipeline and deployment. Use an account with SageMaker training and endpoint quota.

```powershell
$region = "ap-south-1"
$bucket = "YOUR-PRIVATE-BUCKET"
$role = "arn:aws:iam::YOUR-ACCOUNT-ID:role/SageMakerAssignmentRole"
$labPython = ".venv\Scripts\python"
& $labPython pipeline.py --region $region train --bucket $bucket --role $role
```

The `TrainEvaluateRandomForest` step reads two S3 inputs, reports five held-out metrics and publishes `model.tar.gz`. Copy the printed `model_data` URI into the deployment command.

```powershell
$modelData = "s3://YOUR-PRIVATE-BUCKET/path/to/model.tar.gz"
& $labPython pipeline.py --region $region deploy --bucket $bucket --role $role --model-data $modelData
& $labPython pipeline.py --region $region predict
sam deploy --resolve-s3 --stack-name assignment9-sagemaker-api --capabilities CAPABILITY_IAM --region $region
```

Copy the stack's `PredictUrl` output and invoke it with credentials granted `execute-api:Invoke` for `POST /predict` on this API.

```powershell
& $labPython invoke_api.py "YOUR-PREDICT-URL" --region $region
```

The API expects JSON with a `features` array containing 30 numbers in CSV column order. The Lambda role can invoke only the named endpoint.

## Monitoring and cleanup

Review the training job output for elapsed time and billable seconds. In CloudWatch, check the `/aws/sagemaker/TrainingJobs` and `/aws/sagemaker/Endpoints/breast-cancer-rf-endpoint` log groups. Under `AWS/SageMaker`, filter by endpoint and the `AllTraffic` variant, then view `Invocations`, `ModelLatency`, `OverheadLatency`, `Invocation4XXErrors` and `Invocation5XXErrors`. The SAM stack adds a 5XX alarm.

Delete the lab resources after the test. Remove the endpoint first and wait for deletion before removing its configuration and model.

```powershell
sam delete --stack-name assignment9-sagemaker-api --region $region
$endpoint = "breast-cancer-rf-endpoint"
$config = aws sagemaker describe-endpoint --endpoint-name $endpoint --region $region --query EndpointConfigName --output text
$model = aws sagemaker describe-endpoint-config --endpoint-config-name $config --region $region --query "ProductionVariants[0].ModelName" --output text
aws sagemaker delete-endpoint --endpoint-name $endpoint --region $region
aws sagemaker wait endpoint-deleted --endpoint-name $endpoint --region $region
aws sagemaker delete-endpoint-config --endpoint-config-name $config --region $region
aws sagemaker delete-model --model-name $model --region $region
aws sagemaker delete-pipeline --pipeline-name BreastCancerRandomForest --region $region
```

Remove the lab objects from S3 if they are no longer needed. Retain evidence before deleting logs or artifacts.

[SageMaker Pipelines documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html)
