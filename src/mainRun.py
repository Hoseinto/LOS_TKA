import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === Step 1: Encode raw dataset ===
from preprocessing import encode_categorical_variables

raw_data_path = "E:/LOS_TKA/database/raw/TKA_Dataset.csv"
df = pd.read_csv(raw_data_path)
df_encoded = encode_categorical_variables(df)

processed_dir = "E:/LOS_TKA/database/processed"
os.makedirs(processed_dir, exist_ok=True)
encoded_path = os.path.join(processed_dir, "encoded_dataset.csv")
df_encoded.to_csv(encoded_path, index=False)
print(f"✅ Encoded dataset saved to: {encoded_path}")

# === Step 2: Train-test split and scaling ===
X = df_encoded.drop("Prolonged_LOS_Yes", axis=1)
y = df_encoded["Prolonged_LOS_Yes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Step 3: Train and tune models ===
from model_maker import ModelSelector

selector = ModelSelector(X_train_scaled, y_train, save_dir="E:/LOS_TKA/saved_models")
selector.run_all()

# === Step 4: Visualizations ===
from visualizer import ModelVisualizer

visualizer = ModelVisualizer(
    models_dir="E:/LOS_TKA/saved_models",
    X_test=X_test_scaled,
    y_test=y_test,
    output_dir="E:/LOS_TKA/results/figures"
)
visualizer.plot_roc_curves()
visualizer.plot_confusion_matrices()
visualizer.plot_shap_values()

# === Step 5: Evaluation Metrics ===
from model_evaluator import ModelEvaluator

evaluator = ModelEvaluator(
    models_dir="E:/LOS_TKA/saved_models",
    X_test=X_test_scaled,
    y_test=y_test
)

metrics_df = evaluator.evaluate_all()
metrics_dir = "E:/LOS_TKA/results/metrics"
os.makedirs(metrics_dir, exist_ok=True)
metrics_df.to_csv(os.path.join(metrics_dir, "model_metrics.csv"), index=False)

print("📊 Evaluation Summary:")
print(metrics_df)
