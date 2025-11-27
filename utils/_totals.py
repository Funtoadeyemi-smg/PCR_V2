from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


def coerce_total_row(
    df: pd.DataFrame,
    key_column: str,
    numeric_columns: Iterable[str],
) -> pd.DataFrame:
    if key_column not in df.columns:
        return df

    mask = df[key_column].astype(str).str.startswith("Total", na=False)
    if not mask.any():
        total_values = {col: np.nan for col in df.columns}
        total_values[key_column] = "Total"
        for col in numeric_columns:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                total_values[col] = series.sum() if not series.empty else np.nan
        summary_row = pd.DataFrame([total_values], columns=df.columns)
        return pd.concat([df, summary_row], ignore_index=True)

    total_row = df[mask].iloc[-1]
    body = df[~mask].copy()

    def safe_sum(col: str) -> float:
        if col not in body.columns:
            return np.nan
        series = pd.to_numeric(body[col], errors="coerce").dropna()
        return series.sum() if not series.empty else np.nan

    numeric_columns = list(numeric_columns)
    sums = {col: safe_sum(col) for col in numeric_columns}

    match = True
    for col, summed_value in sums.items():
        if col not in total_row or np.isnan(summed_value):
            continue
        try:
            total_value = float(total_row[col])
        except (ValueError, TypeError):
            match = False
            break
        if not np.isclose(total_value, summed_value, rtol=0.01, atol=1e-2):
            match = False
            break

    if not match:
        return df

    summary_values = {
        col: total_row[col] if col in total_row.index else sums.get(col)
        for col in df.columns
    }
    summary_row = pd.DataFrame([summary_values], columns=df.columns)
    return summary_row.reset_index(drop=True)
