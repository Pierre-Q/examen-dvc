import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import json

def evaluate():
    X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
    y_test = pd.read_csv("data/processed_data/y_test.csv").values.ravel()
    model = joblib.load("models/trained_model.pkl")

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    pd.DataFrame(predictions, columns=['prediction']).to_csv("data/predictions.csv", index=False)

    metrics = {
        "mse": mse,
        "r2": r2
    }
    
    with open("metrics/scores.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    evaluate()