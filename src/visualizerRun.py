import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from visualizer import ModelVisualizer

# === Load encoded dataset ===
df = pd.read_csv("E:/LOS_TKA/database/processed/encoded_dataset.csv")
X = df.drop("Prolonged_LOS_Yes", axis=1)
y = df["Prolonged_LOS_Yes"]

# === Split and scale ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_test_scaled = scaler.fit(X_train).transform(X_test)

# === Run visualization ===
visualizer = ModelVisualizer(models_dir="E:/LOS_TKA/saved_models", X_test=X_test_scaled, y_test=y_test)
visualizer.plot_roc_curves()
visualizer.plot_confusion_matrices()
visualizer.plot_shap_values()