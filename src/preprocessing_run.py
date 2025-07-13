import pandas as pd
from preprocessing import encode_categorical_variables
import os

df = pd.read_csv("E:/LOS_TKA/database/raw/TKA_Dataset.csv")
df_encoded = encode_categorical_variables(df)

output_dir = "E:/LOS_TKA/database/processed"
output_filename = "encoded_dataset.csv"
output_path = os.path.join(output_dir, output_filename)
os.makedirs(output_dir, exist_ok=True)
df_encoded.to_csv(output_path, index=False)

print(f"Encoded dataset saved to: {output_path}")
