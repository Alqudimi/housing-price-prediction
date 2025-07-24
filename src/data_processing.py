
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import pickle
def load_and_process_data(file_path):
    df = pd.read_csv(file_path)

    X = df.drop("price", axis=1)
    y = df["price"]

    categorical_features = X.select_dtypes(include=["object"]).columns
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ])

    X_processed = preprocessor.fit_transform(X)

    feature_names = preprocessor.get_feature_names_out()

    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    preprocessor_save_path = 'D:/Programming/python/The_Machine_Learning/housing-price-prediction/models/preprocessor.joblib'
    joblib.dump(preprocessor,preprocessor_save_path)
    
    return X_processed_df, y

if __name__ == "__main__":
    input_file = "../data/raw/Housing_Price_Data_clean.csv"
    output_X_file = "../data/processed/X_processed.csv"
    output_y_file = "../data/processed/y.csv"

    X_processed, y = load_and_process_data(input_file)

    X_processed.to_csv(output_X_file, index=False)
    y.to_csv(output_y_file, index=False)
    print(f"Processed X data saved to {output_X_file}")
    print(f"Target y data saved to {output_y_file}")


