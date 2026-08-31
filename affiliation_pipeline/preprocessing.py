from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping

import pandas as pd

def load_csv(
    path: str | Path,
    *,
    sep: str = ";",
    encoding: str = "utf-8",
    header: int | Literal["infer"] | None = "infer",
    **kwargs,
) -> pd.DataFrame:
    """Load a CSV file with project defaults while allowing pandas options to be overridden."""
    return pd.read_csv(
        path,
        sep=sep,
        encoding=encoding,
        header=header,
        **kwargs,
    )


def preview_dataframe(
    df: pd.DataFrame,
    *,
    n: int = 5,
    random_state: int | None = None,
) -> None:
    """Display column names and a random sample of rows."""
    if n < 1:
        raise ValueError("n must be at least 1.")

    from IPython.display import display

    print("Columns:")
    display(
        pd.DataFrame(
            {
                "position": range(len(df.columns)),
                "column": list(df.columns),
            }
        )
    )

    if df.empty:
        print("DataFrame is empty.")
        return

    display(
        df.sample(
            n=min(n, len(df)),
            random_state=random_state,
        )
    )


def rename_columns(
    df: pd.DataFrame,
    columns: Mapping[object, str],
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Rename one or more columns and return a new DataFrame."""
    if not columns:
        return df.copy()

    missing = [
        source
        for source in columns
        if source not in df.columns
    ]

    if missing and strict:
        joined = ", ".join(repr(column) for column in missing)
        raise KeyError(f"Column(s) do not exist: {joined}.")

    applicable = {
        source: target
        for source, target in columns.items()
        if source in df.columns
    }

    result_columns = [
        applicable.get(column, column)
        for column in df.columns
    ]

    duplicates = pd.Index(result_columns)[
        pd.Index(result_columns).duplicated()
    ].unique()

    if len(duplicates):
        joined = ", ".join(repr(column) for column in duplicates)
        raise ValueError(
            f"Renaming would create duplicate column name(s): {joined}."
        )

    return df.rename(columns=applicable).copy()


def split_column(
    df: pd.DataFrame,
    column: str,
    delimiter: str,
    *,
    into: Literal["rows", "columns"],
    output_column: str | None = None,
    output_prefix: str | None = None,
    row_id_column: str | None = "original_row_id",
    drop_empty: bool = True,
) -> pd.DataFrame:
    """Split one column into new rows or new columns without changing the source column."""
    if column not in df.columns:
        raise KeyError(f"Column {column!r} does not exist.")

    if not delimiter:
        raise ValueError("delimiter must not be empty.")

    if into not in {"rows", "columns"}:
        raise ValueError("into must be either 'rows' or 'columns'.")

    result = df.copy()

    split_values = result[column].astype("string").str.split(
        delimiter,
        regex=False,
    )

    if into == "rows":
        if output_column is None:
            raise ValueError(
                "output_column is required when splitting into rows."
            )

        if output_column == column:
            raise ValueError(
                "output_column must differ from the source column."
            )

        _ensure_columns_do_not_exist(result, [output_column])

        if row_id_column is not None:
            _ensure_columns_do_not_exist(result, [row_id_column])
            result[row_id_column] = result.index

        result[output_column] = split_values
        result = result.explode(output_column, ignore_index=True)
        result[output_column] = result[output_column].astype("string").str.strip()

        if drop_empty:
            result = result[
                result[output_column].notna()
                & result[output_column].ne("")
            ]

        return result.reset_index(drop=True)

    if output_prefix is None:
        raise ValueError(
            "output_prefix is required when splitting into columns."
        )

    cleaned_values = split_values.map(
        lambda parts: _clean_parts(parts, drop_empty=drop_empty)
    )

    max_parts = int(cleaned_values.map(len).max()) if len(result) else 0

    if max_parts == 0:
        return result

    output_columns = [
        f"{output_prefix}_{position}"
        for position in range(1, max_parts + 1)
    ]
    _ensure_columns_do_not_exist(result, output_columns)

    split_frame = pd.DataFrame(
        cleaned_values.tolist(),
        index=result.index,
        columns=range(max_parts),
    )
    split_frame.columns = output_columns
    split_frame = split_frame.astype("string")

    return pd.concat([result, split_frame], axis=1)


def _clean_parts(
    parts: object,
    *,
    drop_empty: bool,
) -> list[str]:
    if not isinstance(parts, list):
        return []

    cleaned = [str(part).strip() for part in parts if pd.notna(part)]

    if drop_empty:
        cleaned = [part for part in cleaned if part]

    return cleaned


def _ensure_columns_do_not_exist(
    df: pd.DataFrame,
    columns: list[str],
) -> None:
    existing = [column for column in columns if column in df.columns]

    if existing:
        joined = ", ".join(repr(column) for column in existing)
        raise ValueError(f"Output column(s) already exist: {joined}.")



def save_csv(
    df: pd.DataFrame,
    filename: str | Path,
    *,
    output_dir: str | Path | None = None,
    sep: str = ";",
    encoding: str = "utf-8",
    index: bool = False,
    overwrite: bool = False,
    **kwargs,
) -> Path:
    """Save a DataFrame as CSV using the project defaults."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "data"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(filename)

    if filename.suffix.lower() != ".csv":
        filename = filename.with_suffix(".csv")

    path = filename if filename.is_absolute() else output_dir / filename

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {path}. "
            "Set overwrite=True to replace it."
        )

    df.to_csv(
        path,
        sep=sep,
        encoding=encoding,
        index=index,
        **kwargs,
    )

    return path

