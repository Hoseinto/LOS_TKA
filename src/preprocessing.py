import pandas as pd

def encode_categorical_variables(df, drop_first=True):
    """
    Encodes categorical variables in a DataFrame using one-hot encoding.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        drop_first (bool): Whether to drop the first category to avoid multicollinearity.

    Returns:
        pd.DataFrame: DataFrame with categorical variables one-hot encoded.
    """
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)
    return df_encoded
