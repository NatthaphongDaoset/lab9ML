import mlflow
from sklearn.datasets import load_breast_cancer


def load_and_predict():
    MODEL_NAME = "cancer-classifier-prod"
    MODEL_ALIAS = "staging"

    print(f"Loading model '{MODEL_NAME}' with alias '@{MODEL_ALIAS}'...")

    try:
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        )
    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        return

    # โหลดข้อมูล
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    target_names = load_breast_cancer().target_names

    # หยิบตัวอย่างรายแรกของแต่ละคลาส
    idx_malignant = y[y == 0].index[0]   # คลาส 0 = malignant
    idx_benign = y[y == 1].index[0]      # คลาส 1 = benign

    samples = X.loc[[idx_malignant, idx_benign]]
    actuals = y.loc[[idx_malignant, idx_benign]]

    predictions = model.predict(samples)

    print("-" * 40)
    for i, (actual, pred) in enumerate(zip(actuals, predictions)):
        actual_name = target_names[actual]
        pred_name = target_names[pred]
        correct = "✅ ถูก" if actual == pred else "❌ ผิด"
        print(f"ตัวอย่างที่ {i+1}:")
        print(f"  Actual    : {actual_name}")
        print(f"  Predicted : {pred_name}")
        print(f"  ผล        : {correct}")
        print()
    print("-" * 40)


if __name__ == "__main__":
    load_and_predict()