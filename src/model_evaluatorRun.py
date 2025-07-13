import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model_evaluator import ModelEvaluator

# === Load dataset ===
df = pd.read_csv("E:/LOS_TKA/database/processed/encoded_dataset.csv")
X = df.drop("Prolonged_LOS_Yes", axis=1)
y = df["Prolonged_LOS_Yes"]

# === Split and scale ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_test_scaled = scaler.fit(X_train).transform(X_test)

# === Evaluate models ===
evaluator = ModelEvaluator(models_dir="saved_models", X_test=X_test_scaled, y_test=y_test)
results_df = evaluator.evaluate_all()

# === Save results ===
results_dir = "results/metrics"
os.makedirs(results_dir, exist_ok=True)
results_df.to_csv(f"{results_dir}/model_metrics.csv", index=False)

# === Print summary ===
print(results_df)
