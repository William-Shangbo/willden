from __future__ import annotations

from typing import Iterable

import pandas as pd


def ewma_lambda(half_life: int) -> float:
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    return 0.5 ** (1.0 / half_life)


def causal_ewma(
    df: pd.DataFrame,
    value_cols: Iterable[str],
    group_cols: list[str],
    order_cols: list[str],
    half_life: int,
    min_count: int = 1,
) -> pd.DataFrame:
    """
    Compute causal EWMA within each group after sorting by `order_cols`.
    The current row is allowed to use its current observed value.
    """
    lambda_ = ewma_lambda(half_life)
    alpha = 1.0 - lambda_
    value_cols = list(value_cols)
    ordered = df.sort_values(group_cols + order_cols).copy()

    def _transform_group(group: pd.DataFrame) -> pd.DataFrame:
        values = group[value_cols]
        ewma_df = values.ewm(
            alpha=alpha,
            adjust=False,
            min_periods=min_count,
            ignore_na=True,
        ).mean()
        return ewma_df

    ewma_df = (
        ordered.groupby(group_cols, group_keys=False, sort=False)
        .apply(_transform_group)
        .reset_index(drop=True)
    )
    ewma_df.index = ordered.index
    return ewma_df.sort_index()


def causal_ewma_impute(
    df: pd.DataFrame,
    value_cols: Iterable[str],
    group_cols: list[str],
    order_cols: list[str],
    half_life: int,
    min_count: int = 1,
) -> pd.DataFrame:
    """
    Impute missing values using causal EWMA from current and past observations.
    Observed values are preserved. Missing values are replaced by the EWMA state.
    """
    value_cols = list(value_cols)
    ewma_df = causal_ewma(
        df=df,
        value_cols=value_cols,
        group_cols=group_cols,
        order_cols=order_cols,
        half_life=half_life,
        min_count=min_count,
    )
    original = df[value_cols]
    return original.where(original.notna(), ewma_df)
