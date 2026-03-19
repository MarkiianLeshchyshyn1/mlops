from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset for training.")
    parser.add_argument("input_file", type=Path, help="Path to raw CSV dataset.")
    parser.add_argument(
        "output_dir", type=Path, help="Directory for prepared train/test CSV files."
    )
    parser.add_argument("--text-column", type=str, default="review")
    parser.add_argument("--target-column", type=str, default="sentiment")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, text_column: str, target_column: str) -> None:
    required = {text_column, target_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def clean_dataset(
    df: pd.DataFrame, text_column: str, target_column: str
) -> pd.DataFrame:
    cleaned = df[[text_column, target_column]].dropna()
    cleaned[text_column] = cleaned[text_column].astype(str).str.strip()
    cleaned[target_column] = cleaned[target_column].astype(str).str.strip().str.lower()
    cleaned = cleaned[cleaned[text_column] != ""]
    cleaned = cleaned[cleaned[target_column].isin(["negative", "positive"])]
    return cleaned.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if not args.input_file.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_file, low_memory=False)
    validate_columns(df, args.text_column, args.target_column)
    prepared = clean_dataset(df, args.text_column, args.target_column)

    train_df, test_df = train_test_split(
        prepared,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=prepared[args.target_column],
    )

    train_path = args.output_dir / "train.csv"
    test_path = args.output_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Prepared rows total: {len(prepared)}")
    print(f"Saved train: {train_path} ({len(train_df)} rows)")
    print(f"Saved test: {test_path} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()
