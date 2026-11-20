"""Train and evaluate the same Random Forest locally or on SageMaker."""

import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRICS = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "auc": roc_auc_score,
}


def model_fn(model_dir):
    """Load the trained artifact in the SageMaker scikit-learn container."""
    return joblib.load(Path(model_dir) / "model.joblib")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--train", type=Path, default=os.getenv("SM_CHANNEL_TRAIN", "data"))
    parser.add_argument("--test", type=Path, default=os.getenv("SM_CHANNEL_TEST", "data"))
    parser.add_argument("--model-dir", type=Path, default=os.getenv("SM_MODEL_DIR", "artifacts"))
    args = parser.parse_args()

    train = pd.read_csv(args.train / "train.csv")
    test = pd.read_csv(args.test / "test.csv")
    features = train.drop(columns="target").columns
    model = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42
    )
    model.fit(train[features].to_numpy(), train["target"])
    test_features = test[features].to_numpy()
    predictions = model.predict(test_features)
    probabilities = model.predict_proba(test_features)[:, 1]

    print(f"train_rows={len(train)} test_rows={len(test)} features={len(features)}")
    print(f"n_estimators={args.n_estimators} max_depth={args.max_depth}")
    for name, metric in METRICS.items():
        values = probabilities if name == "auc" else predictions
        print(f"test_{name}={metric(test['target'], values):.4f}")
    matrix = confusion_matrix(test["target"], predictions, labels=[0, 1])
    print(f"confusion_matrix [0=malignant, 1=benign]: {matrix.tolist()}")
    print(f"first_5_predictions={predictions[:5].tolist()}")
    print(f"first_5_actual={test['target'].iloc[:5].tolist()}")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_dir / "model.joblib")


if __name__ == "__main__":
    main()
