"""
Data loading, feature engineering, train/val/test splitting, and SMOTE.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_data(path="data/creditcard.csv"):
    """Load the credit card fraud dataset from CSV."""
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Class distribution:\n{df['Class'].value_counts()}")
    print(f"Fraud ratio: {df['Class'].mean():.4%}")
    return df


def preprocess(df):
    """
    Feature engineering:
    - Scale Amount with RobustScaler
    - Convert Time to Hour of day
    - Drop original Time column

    Returns preprocessed DataFrame and list of feature names.
    """
    df = df.copy()

    # Scale Amount
    scaler = RobustScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    # Time -> Hour of day
    df["Hour"] = (df["Time"] % 86400) / 3600
    df = df.drop("Time", axis=1)

    feature_names = [c for c in df.columns if c != "Class"]
    print(f"Features after preprocessing ({len(feature_names)}): {feature_names}")
    return df, feature_names


def split_data(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Stratified train/validation/test split.

    Returns (X_train, X_val, X_test, y_train, y_val, y_test) as numpy arrays.
    """
    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].values
    y = df["Class"].values

    # First split: train vs temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_ratio + test_ratio), stratify=y, random_state=random_state
    )

    # Second split: val vs test (50/50 of temp)
    val_frac = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_frac), stratify=y_temp, random_state=random_state
    )

    print(f"Train: {X_train.shape[0]} (fraud: {y_train.sum()})")
    print(f"Val:   {X_val.shape[0]} (fraud: {y_val.sum()})")
    print(f"Test:  {X_test.shape[0]} (fraud: {y_test.sum()})")

    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X_train, y_train, sampling_strategy=0.2, random_state=42):
    """
    Apply SMOTE to the training set only.

    sampling_strategy=0.2 means minority class will be 20% of majority class
    (i.e., 1:5 ratio).

    Returns resampled (X_train_sm, y_train_sm).
    """
    print(f"Before SMOTE: {X_train.shape[0]} samples, fraud: {y_train.sum()}")

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE:  {X_train_sm.shape[0]} samples, fraud: {y_train_sm.sum()}")
    return X_train_sm, y_train_sm
