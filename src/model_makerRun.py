import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model_maker import ModelSelector  

df = pd.read_csv("E:/LOS_TKA/database/processed/encoded_dataset.csv")  

X = df.drop("Prolonged_LOS_Yes", axis=1)
y = df["Prolonged_LOS_Yes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

selector = ModelSelector(X_train_scaled, y_train, save_dir="saved_models")
selector.run_all()
