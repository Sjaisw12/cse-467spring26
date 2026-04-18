#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income_group",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean Adult dataset from archive.zip and split into client Excel files."
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        required=True,
        help="Path to archive.zip containing adult-training.csv and adult-test.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("client_output"),
        help="Directory where cleaned files and client Excel files will be saved",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=10,
        help="Number of clients to create (default: 10)",
    )
    parser.add_argument(
        "--rows-per-client",
        type=int,
        default=None,
        help=(
            "Optional fixed number of rows per client. "
            "If omitted, the script uses the whole cleaned dataset and splits it equally."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling before splitting (default: 42)",
    )
    return parser.parse_args()


def read_adult_from_zip(zip_path: Path) -> pd.DataFrame:
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    with ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        required = {"adult-training.csv", "adult-test.csv"}
        missing = required - names
        if missing:
            raise FileNotFoundError(
                f"Missing expected files inside ZIP: {sorted(missing)}"
            )

        train_df = pd.read_csv(
            zf.open("adult-training.csv"),
            header=None,
            names=ADULT_COLUMNS,
            skipinitialspace=True,
        )
        test_df = pd.read_csv(
            zf.open("adult-test.csv"),
            header=None,
            names=ADULT_COLUMNS,
            skiprows=1,  # first line is metadata: |1x3 Cross validator
            skipinitialspace=True,
        )

    return pd.concat([train_df, test_df], ignore_index=True)


def add_age_group(age: int) -> str:
    if age < 25:
        return "18-24"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    if age < 65:
        return "55-64"
    return "65+"


def clean_adult(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    # Normalize whitespace for all string columns.
    for col in cleaned.select_dtypes(include="object").columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    # Standardize target labels in test/train combined data.
    cleaned["income_group"] = cleaned["income_group"].str.replace(".", "", regex=False)
    cleaned["income_group"] = cleaned["income_group"].replace({"<=50K": "low", ">50K": "high"})

    # Replace unknown markers with NA, then drop incomplete rows.
    cleaned = cleaned.replace("?", pd.NA)
    cleaned = cleaned.dropna().reset_index(drop=True)

    # Convert numeric columns.
    numeric_cols = [
        "age",
        "fnlwgt",
        "education_num",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
    ]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned = cleaned.dropna().reset_index(drop=True)

    # Add helper column for your project plan.
    cleaned["age_group"] = cleaned["age"].apply(add_age_group)

    # Reorder columns to place project-relevant fields first.
    preferred_order = [
        "age",
        "age_group",
        "gender",
        "race",
        "income_group",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
    ]
    cleaned = cleaned[preferred_order]

    return cleaned


def write_clients(df: pd.DataFrame, output_dir: Path, num_clients: int, rows_per_client: int | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if rows_per_client is not None:
        required_rows = num_clients * rows_per_client
        if len(df) < required_rows:
            raise ValueError(
                f"Not enough cleaned rows ({len(df)}) for {num_clients} clients x {rows_per_client} rows each."
            )
        working_df = df.iloc[:required_rows].copy()
        client_splits = [
            working_df.iloc[i * rows_per_client : (i + 1) * rows_per_client].copy()
            for i in range(num_clients)
        ]
        leftovers = df.iloc[required_rows:].copy()
    else:
        working_df = df.copy()
        base_size = len(working_df) // num_clients
        remainder = len(working_df) % num_clients
        client_splits = []
        start = 0
        for i in range(num_clients):
            extra = 1 if i < remainder else 0
            end = start + base_size + extra
            client_splits.append(working_df.iloc[start:end].copy())
            start = end
        leftovers = pd.DataFrame(columns=working_df.columns)

    # Save each client to Excel.
    for idx, client_df in enumerate(client_splits, start=1):
        client_path = output_dir / f"client_{idx}.xlsx"
        client_df.to_excel(client_path, index=False)

    if not leftovers.empty:
        leftovers.to_excel(output_dir / "leftover_rows.xlsx", index=False)

    # Summary file.
    summary = pd.DataFrame(
        {
            "client_id": [f"client_{i}" for i in range(1, len(client_splits) + 1)],
            "rows": [len(split) for split in client_splits],
        }
    )
    summary.to_csv(output_dir / "client_split_summary.csv", index=False)


def main() -> None:
    args = parse_args()

    raw_df = read_adult_from_zip(args.zip_path)
    cleaned_df = clean_adult(raw_df)

    # Shuffle once before splitting so clients are not ordered by original file position.
    cleaned_df = cleaned_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(args.output_dir / "adult_cleaned_full.csv", index=False)
    cleaned_df.to_excel(args.output_dir / "adult_cleaned_full.xlsx", index=False)

    write_clients(
        df=cleaned_df,
        output_dir=args.output_dir,
        num_clients=args.num_clients,
        rows_per_client=args.rows_per_client,
    )

    print("Done.")
    print(f"Cleaned full dataset rows: {len(cleaned_df)}")
    print(f"Output folder: {args.output_dir.resolve()}")
    if args.rows_per_client is None:
        print(f"Split mode: full dataset equally across {args.num_clients} clients")
    else:
        print(
            f"Split mode: {args.num_clients} clients x {args.rows_per_client} rows each "
            f"(leftovers, if any, saved separately)"
        )


if __name__ == "__main__":
    main()
