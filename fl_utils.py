"""
fl_utils.py

Shared utilities for federated learning pipeline.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


TARGET_COLUMN = "income_group"

SENSITIVE_CANDIDATES = [
    "sex",
    "gender",
    "race",
    "age_group",
]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().str.lower()

    return df


def find_target_column(df: pd.DataFrame) -> str:
    if TARGET_COLUMN in df.columns:
        return TARGET_COLUMN

    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found. Available columns: {list(df.columns)}"
    )


def find_sensitive_column(df: pd.DataFrame) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}

    for candidate in SENSITIVE_CANDIDATES:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def normalize_income_group(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()

    mapping = {
        "low": 0,
        "medium": 1,
        "high": 2,
        "0": 0,
        "1": 1,
        "2": 2,
    }

    return s.map(mapping).astype(int)


def build_feature_schema(client_paths: List[str]) -> Tuple[List[str], str]:
    target_column = TARGET_COLUMN
    all_feature_columns = set()

    for path in client_paths:
        df = pd.read_excel(path)
        df = clean_dataframe(df)

        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in {path}. "
                f"Available columns: {list(df.columns)}"
            )

        X = df.drop(columns=[target_column])
        X_encoded = pd.get_dummies(X, drop_first=False)

        all_feature_columns.update(X_encoded.columns)

    feature_columns = sorted(list(all_feature_columns))

    return feature_columns, target_column


def load_client_dataset(
    data_path: str,
    feature_columns: List[str],
    target_column: Optional[str] = None,
):
    df = pd.read_excel(data_path)
    df = clean_dataframe(df)

    if target_column is None:
        target_column = TARGET_COLUMN

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in {data_path}. "
            f"Available columns: {list(df.columns)}"
        )

    sensitive_column = find_sensitive_column(df)

    y = normalize_income_group(df[target_column])

    X_raw = df.drop(columns=[target_column])
    X_encoded = pd.get_dummies(X_raw, drop_first=False)
    X_encoded = X_encoded.reindex(columns=feature_columns, fill_value=0)

    sensitive_values = None
    if sensitive_column is not None:
        sensitive_values = df[sensitive_column].copy()

    return (
        X_encoded.to_numpy(dtype=np.float64),
        y.to_numpy(dtype=np.int64),
        sensitive_values,
        target_column,
        sensitive_column,
    )