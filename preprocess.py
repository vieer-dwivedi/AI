import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Input & output paths
RAW_DATA_PATH = "data/raw/bank-full.csv"
PROCESSED_PATH = "data/processed/"

def load_data(path=RAW_DATA_PATH):
    """Load raw dataset"""
    df = pd.read_csv(path, sep=";")  # UCI dataset is semicolon-separated
    return df

def clean_data(df):
    """Basic cleaning + encoding"""
    # Drop duplicates
    df = df.drop_duplicates()

    # Encode categorical features
    categorical_cols = df.select_dtypes(include=["object"]).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders

def split_data(df):
    """Split into train, validation, test"""
    train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df["y"])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["y"])
    return train, val, test

def save_splits(train, val, test, out_dir=PROCESSED_PATH):
    train.to_csv(out_dir + "train.csv", index=False)
    val.to_csv(out_dir + "val.csv", index=False)
    test.to_csv(out_dir + "test.csv", index=False)
    print(f"✅ Data saved in {out_dir}")

if __name__ == "__main__":
    df = load_data()
    df_clean, encoders = clean_data(df)
    train, val, test = split_data(df_clean)
    save_splits(train, val, test)
