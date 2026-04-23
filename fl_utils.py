"""
fl_utils.py

Shared utilities for federated learning pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


TARGET_CANDIDATES = [
    "income",
    "label",
    "target",
    "class",
    "y",
    "salary"
]

SENSITIVE_CANDIDATES = [
    "sex",
    "gender",
    "race"
]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    return df


def find_target_column(df: pd.DataFrame) -> str:
    lower_map = {c.lower(): c for c in df.columns}

    for candidate in TARGET_CANDIDATES:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    # fallback: use last column
    return df.columns[-1]


def find_sensitive_column(df: pd.DataFrame) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}

    for candidate in SENSITIVE_CANDIDATES:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def normalize_binary_label(series: pd.Series) -> pd.Series:
    """
    Converts common Adult-income style labels into 0/1.
    """
    s = series.copy()

    if pd.api.types.is_numeric_dtype(s):
        unique_vals = sorted(pd.Series(s).dropna().unique())
        if set(unique_vals).issubset({0, 1}):
            return s.astype(int)

    s = s.astype(str).str.strip().str.replace(".", "", regex=False)

    positive_values = {
        ">50K", "1", "yes", "true", "positive", "high", "approved"
    }

    return s.apply(lambda x: 1 if x in positive_values else 0).astype(int)


def build_feature_schema(client_paths: List[str]) -> Tuple[List[str], str]:
    """
    Build one shared feature schema across all clients so dummy columns align.
    Returns:
        feature_columns, target_column
    """
    all_feature_frames = []
    detected_target = None

    for path in client_paths:
        df = pd.read_excel(path)
        df = clean_dataframe(df)

        target_col = find_target_column(df)
        if detected_target is None:
            detected_target = target_col

        X = df.drop(columns=[target_col], errors="ignore")
        X_encoded = pd.get_dummies(X, drop_first=False)
        all_feature_frames.append(X_encoded)

    combined = pd.concat(all_feature_frames, axis=0, ignore_index=True).fillna(0)
    feature_columns = list(combined.columns)

    return feature_columns, detected_target


def load_client_dataset(
    data_path: str,
    feature_columns: List[str],
    target_column: Optional[str] = None,
):
    df = pd.read_excel(data_path)
    df = clean_dataframe(df)

    if target_column is None:
        target_column = find_target_column(df)

    sensitive_column = find_sensitive_column(df)

    y = normalize_binary_label(df[target_column])

    X_raw = df.drop(columns=[target_column], errors="ignore")
    X_encoded = pd.get_dummies(X_raw, drop_first=False)

    X_encoded = X_encoded.reindex(columns=feature_columns, fill_value=0)

    sensitive_values = None
    if sensitive_column is not None and sensitive_column in df.columns:
        sensitive_values = df[sensitive_column].copy()

    return (
        X_encoded.to_numpy(dtype=np.float64),
        y.to_numpy(dtype=np.int64),
        sensitive_values,
        target_column,
        sensitive_column,
    )