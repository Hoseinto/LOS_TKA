import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    log_loss
)

class ModelEvaluator:
    def __init__(self, models_dir, X_test, y_test):
        self.models_dir = models_dir
        self.X_test = X_test
        self.y_test = y_test
        self.models = self._load_models()

    def _load_models(self):
        models = {}
        for file in os.listdir(self.models_dir):
            if file.endswith("_best.joblib"):
                name = file.replace("_best.joblib", "")
                models[name] = joblib.load(os.path.join(self.models_dir, file))
        return models

    def evaluate_all(self):
        rows = []
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, "predict_proba") else None
            tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()

            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            ll = log_loss(self.y_test, y_proba) if y_proba is not None else np.nan

            rows.append({
                "Model": name,
                "Precision": round(precision, 3),
                "Recall": round(recall, 3),
                "F1-Score": round(f1, 3),
                "Specificity": round(specificity, 3),
                "NPV": round(npv, 3),
                "Log Loss": round(ll, 3)
            })

        results_df = pd.DataFrame(rows).sort_values("F1-Score", ascending=False)
        return results_df
