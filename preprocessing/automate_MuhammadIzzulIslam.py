"""
automate_MuhammadIzzulIslam.py

Script ini mengotomatisasi seluruh tahapan data preprocessing yang telah
dilakukan secara manual pada notebook `Eksperimen_MuhammadIzzulIslam.ipynb`.

Tahapan yang dilakukan:
1. Memuat dataset mentah
2. Menghapus kolom yang tidak relevan
3. Menangani missing values
4. Menghapus data duplikat
5. Parsing kolom Credit_History_Age menjadi numerik
6. Encoding fitur kategorikal dan target
7. Standarisasi fitur numerik
8. Split data menjadi train dan test set
9. Menyimpan hasil preprocessing ke folder output

Penggunaan:
    python automate_MuhammadIzzulIslam.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def parse_credit_history_age(text):
    """Mengubah teks 'X Years and Y Months' menjadi total bulan (numerik)."""
    try:
        years = int(text.split(' Years')[0])
        months = int(text.split('and ')[1].split(' Months')[0])
        return years * 12 + months
    except Exception:
        return np.nan


def load_data(raw_path: str) -> pd.DataFrame:
    """Memuat dataset mentah dari path yang diberikan."""
    df = pd.read_csv(raw_path)
    print(f"[INFO] Dataset berhasil dimuat: {df.shape}")
    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus kolom yang tidak relevan untuk pemodelan."""
    cols_to_drop = [c for c in ['ID', 'Customer_ID', 'Name', 'SSN', 'Month'] if c in df.columns]
    df_clean = df.drop(columns=cols_to_drop)
    print(f"[INFO] Kolom dihapus: {cols_to_drop}")
    return df_clean


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Menangani missing values pada kolom numerik dengan imputasi median."""
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    print(f"[INFO] Sisa missing values setelah imputasi: {df.isnull().sum().sum()}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus baris duplikat."""
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"[INFO] Baris duplikat dihapus: {before - after}")
    return df


def transform_credit_history_age(df: pd.DataFrame) -> pd.DataFrame:
    """Mengonversi kolom Credit_History_Age menjadi fitur numerik dalam bulan."""
    if 'Credit_History_Age' in df.columns:
        df['Credit_History_Age_Months'] = df['Credit_History_Age'].apply(parse_credit_history_age)
        df = df.drop(columns=['Credit_History_Age'])
        df['Credit_History_Age_Months'] = df['Credit_History_Age_Months'].fillna(
            df['Credit_History_Age_Months'].median()
        )
    return df


def encode_features(df: pd.DataFrame, target_col: str = 'Credit_Score'):
    """Melakukan encoding pada fitur kategorikal dan target."""
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    target_encoder = LabelEncoder()
    df[target_col] = target_encoder.fit_transform(df[target_col])

    print(f"[INFO] Kolom kategorikal diencode: {categorical_cols}")
    print(f"[INFO] Mapping target: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")

    return df, label_encoders, target_encoder


def scale_features(df: pd.DataFrame, target_col: str = 'Credit_Score'):
    """Melakukan standarisasi pada seluruh fitur numerik (kecuali target)."""
    feature_cols = [c for c in df.columns if c != target_col]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler, feature_cols


def split_and_save(df: pd.DataFrame, feature_cols: list, target_col: str,
                    output_dir: str, test_size: float = 0.2, random_state: int = 42):
    """Membagi data menjadi train/test set dan menyimpan hasilnya."""
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    os.makedirs(output_dir, exist_ok=True)

    train_final = X_train.copy()
    train_final[target_col] = y_train.values
    test_final = X_test.copy()
    test_final[target_col] = y_test.values

    train_path = os.path.join(output_dir, 'train_data.csv')
    test_path = os.path.join(output_dir, 'test_data.csv')

    train_final.to_csv(train_path, index=False)
    test_final.to_csv(test_path, index=False)

    print(f"[INFO] Train data disimpan di: {train_path} ({train_final.shape})")
    print(f"[INFO] Test data disimpan di: {test_path} ({test_final.shape})")

    return train_final, test_final


def preprocess_data(raw_path: str, output_dir: str) -> tuple:
    """
    Fungsi utama yang menjalankan seluruh pipeline preprocessing secara otomatis
    dan mengembalikan data yang siap dilatih.

    Args:
        raw_path (str): Path ke file dataset mentah (CSV).
        output_dir (str): Path folder untuk menyimpan hasil preprocessing.

    Returns:
        tuple: (train_df, test_df) - DataFrame hasil preprocessing yang siap dilatih.
    """
    df = load_data(raw_path)
    df = drop_irrelevant_columns(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = transform_credit_history_age(df)
    df, label_encoders, target_encoder = encode_features(df)
    df, scaler, feature_cols = scale_features(df)
    train_df, test_df = split_and_save(df, feature_cols, 'Credit_Score', output_dir)

    print("[SUCCESS] Pipeline preprocessing selesai dijalankan.")
    return train_df, test_df


if __name__ == '__main__':
    RAW_DATA_PATH = '../credit_scoring_raw/credit_score_data.csv'
    OUTPUT_DIR = 'credit_scoring_preprocessing'

    preprocess_data(RAW_DATA_PATH, OUTPUT_DIR)
