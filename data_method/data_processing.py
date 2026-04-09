import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'file_method'))
from file_management import read_byclass

TIMEIDS_PER_DAY = 239
MAX_HISTORICAL_TIMESTAMPS = 2 * TIMEIDS_PER_DAY


def _weight_transform(values, weights):
    """
    对每一列乘上 weight，weight 可能存在的情况：数值 or nan
    当 weight = nan 的时候，impute weight by average of non-missing weight
    当 value = nan 时，返回 nan
    
    Args:
        values: 特征值数组
        weights: 权重数组
    
    Returns:
        values * weights
    """
    result = np.full(len(values), np.nan)
    
    non_nan_mask = ~pd.isna(values)
    non_nan_values = values[non_nan_mask]
    
    if len(non_nan_values) == 0:
        return result
    
    weight_nan_mask = pd.isna(weights)
    non_nan_weights = weights[~weight_nan_mask]
    
    if len(non_nan_weights) > 0:
        avg_weight = np.mean(non_nan_weights)
    else:
        avg_weight = 1.0
    
    imputed_weights = np.where(weight_nan_mask, avg_weight, weights)
    
    result[non_nan_mask] = non_nan_values * imputed_weights[non_nan_mask]
    
    return result


def _normalize_to_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _build_combined_weights(group_df, weights):
    if not weights:
        return None

    weight_arrays = []
    for weight_col in weights:
        if weight_col in group_df.columns:
            weight_arrays.append(group_df[weight_col].to_numpy(dtype=float, copy=False))

    if not weight_arrays:
        return None

    combined_weights = np.sum(weight_arrays, axis=0)
    weight_nan_mask = pd.isna(combined_weights)
    if weight_nan_mask.any():
        non_nan_weights = combined_weights[~weight_nan_mask]
        avg_weight = np.mean(non_nan_weights) if len(non_nan_weights) > 0 else 1.0
        combined_weights = np.where(weight_nan_mask, avg_weight, combined_weights)

    return combined_weights


def _normalize_rank_values(valid_values):
    if len(valid_values) == 0:
        return np.array([], dtype=float)

    if len(valid_values) == 1:
        return np.array([0.5], dtype=float)

    rank_values = pd.Series(valid_values).rank(method='average').to_numpy(dtype=float)
    return (rank_values - 1.0) / (len(valid_values) - 1.0)


def _get_valid_values_and_weights(group_df, col, weights):
    values = group_df[col].to_numpy(dtype=float, copy=False)
    valid_mask = ~pd.isna(values)

    combined_weights = _build_combined_weights(group_df, weights)
    if combined_weights is not None:
        valid_mask &= ~pd.isna(combined_weights)
        combined_weights = combined_weights[valid_mask]

    valid_values = values[valid_mask]
    return values, valid_mask, valid_values, combined_weights


def weighted_quantile(values, weights, quantile):
    if len(values) == 0:
        return np.nan

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    total_weight = np.sum(sorted_weights)
    if total_weight <= 0 or np.isnan(total_weight):
        return np.nan

    cumsum_weights = np.cumsum(sorted_weights)
    cumsum_weights /= total_weight

    return np.interp(quantile, cumsum_weights, sorted_values)


def _construct_groups(df, by):
    if by:
        groups = df.groupby(by, dropna=False)
    else:
        class SingleGroup:
            def __iter__(self):
                yield (None, df)
        groups = SingleGroup()
    
    return groups


def _rank_unweighted_group(values_df):
    ranked_df = values_df.rank(method='average', na_option='keep')
    count_array = values_df.count().to_numpy(dtype=float)
    count_df = pd.DataFrame(
        np.repeat(count_array[None, :], len(values_df), axis=0),
        index=values_df.index,
        columns=values_df.columns,
    )
    normalized_df = (ranked_df - 1.0) / (count_df - 1.0)

    singleton_mask = count_df.eq(1) & ranked_df.notna()
    normalized_df = normalized_df.mask(singleton_mask, 0.5)

    # Explicitly keep all-NaN groups as NaN instead of allowing unstable fill artifacts.
    all_nan_mask = count_df.eq(0)
    normalized_df = normalized_df.mask(all_nan_mask)
    return normalized_df


def rank(df, columns, by=None, weights=None, epsilon=1e-8):
    """
    计算 columns 的排名，精度为 1e-8
    排名在 [0, 1] 之间均匀分布
    nan 进入 rank 得到 nan
    
    Args:
        df: 要处理的 DataFrame
        columns: 要计算排名的列名，可以是字符串或列表
        by: 分组列名，在每一组内计算排名，如果为 None 则对所有行计算
        weights: 权重列名，在计算排名时使用加权，如果为 None 则直接计算排名
        epsilon: 排名精度，默认 1e-8
    
    Returns:
        包含排名列的 DataFrame，列名为 {原列名}_r
    """
    df = df.copy()
    original_index = df.index.copy()
    n_rows = len(df)

    columns = _normalize_to_list(columns)
    by = _normalize_to_list(by)
    weights = _normalize_to_list(weights)

    existing_columns = [col for col in columns if col in df.columns]
    if not existing_columns:
        return pd.DataFrame(index=original_index)

    if not weights:
        if by:
            groups = _construct_groups(df, by)
            ranked_parts = []
            for _, group_df in groups:
                if group_df.empty:
                    continue
                group_values = group_df[existing_columns]
                ranked_parts.append(_rank_unweighted_group(group_values))
            normalized_df = pd.concat(ranked_parts).reindex(original_index)
        else:
            normalized_df = _rank_unweighted_group(df[existing_columns])
        normalized_df.columns = [f'{col}_r' for col in existing_columns]
        requested_columns = [f'{col}_r' for col in columns]
        result_df = normalized_df.reindex(columns=requested_columns)
        return result_df

    groups = _construct_groups(df, by)

    rank_arrays = {f'{col}_r': np.full(n_rows, np.nan) for col in columns}

    for group_name, group_df in groups:
        if group_df.empty:
            continue

        group_positions = original_index.get_indexer(group_df.index)
        combined_weights = _build_combined_weights(group_df, weights)

        for col in columns:
            if col not in group_df.columns:
                continue

            values = group_df[col].to_numpy(dtype=float, copy=False)
            valid_mask = ~pd.isna(values)
            transformed_values = values

            if combined_weights is not None:
                transformed_values = _weight_transform(values, combined_weights)
                valid_mask = ~pd.isna(transformed_values)

            valid_values = transformed_values[valid_mask]
            if len(valid_values) == 0:
                continue

            normalized_ranks = _normalize_rank_values(valid_values)
            rank_arrays[f'{col}_r'][group_positions[valid_mask]] = normalized_ranks

    result_df = pd.DataFrame(rank_arrays, index=original_index)
    return result_df


def standardize(df, columns, by=None, weights=None, na_action='ignore'):
    df = df.copy()
    original_index = df.index.copy()

    columns = _normalize_to_list(columns)
    by = _normalize_to_list(by)
    weights = _normalize_to_list(weights)

    groups = _construct_groups(df, by)

    zscore_results = {}
    for col in columns:
        zscore_results[f'{col}_zscore'] = pd.Series(np.full(len(df), np.nan), index=df.index)

    for group_name, group_df in groups:
        if group_df.empty:
            continue

        group_indices = group_df.index.values

        for col in columns:
            if col not in group_df.columns:
                continue

            values, valid_mask, valid_values, valid_weights = _get_valid_values_and_weights(group_df, col, weights)
            zscore = np.full(len(values), np.nan, dtype=float)

            if len(valid_values) == 0:
                zscore_results[f'{col}_zscore'].loc[group_indices] = zscore
                continue

            if valid_weights is not None:
                total_weight = np.sum(valid_weights)
                if total_weight <= 0 or np.isnan(total_weight):
                    zscore_results[f'{col}_zscore'].loc[group_indices] = zscore
                    continue

                weighted_mean = np.sum(valid_values * valid_weights) / total_weight
                weighted_variance = np.sum(valid_weights * (valid_values - weighted_mean) ** 2) / total_weight
            else:
                weighted_mean = np.mean(valid_values)
                weighted_variance = np.var(valid_values)

            weighted_std = np.sqrt(weighted_variance)
            if weighted_std > 0:
                zscore[valid_mask] = (valid_values - weighted_mean) / weighted_std

            if na_action == 'ignore':
                zscore[pd.isna(values)] = np.nan
            elif na_action == 'concerned':
                nan_mask = ~valid_mask
                if nan_mask.any():
                    finite_zscore = zscore[~nan_mask]
                    if len(finite_zscore) > 0:
                        avg_zscore = np.mean(finite_zscore)
                    else:
                        avg_zscore = 0.0
                    zscore[nan_mask] = avg_zscore

            zscore_results[f'{col}_zscore'].loc[group_indices] = zscore

    result_df = pd.DataFrame(zscore_results)
    result_df.index = original_index
    return result_df


def winsorize(df, columns, by=None, weights=None, lower=0.05, upper=0.95, na_action='ignore'):
    df = df.copy()
    original_index = df.index.copy()

    columns = _normalize_to_list(columns)
    by = _normalize_to_list(by)
    weights = _normalize_to_list(weights)

    groups = _construct_groups(df, by)

    clipped_results = {}
    for col in columns:
        clipped_results[col] = df[col].copy()

    for group_name, group_df in groups:
        if group_df.empty:
            continue

        group_indices = group_df.index.values

        for col in columns:
            if col not in group_df.columns:
                continue

            original_dtype = group_df[col].dtype
            values, valid_mask, valid_values, valid_weights = _get_valid_values_and_weights(group_df, col, weights)
            if len(valid_values) == 0:
                continue

            if valid_weights is not None:
                lower_bound = weighted_quantile(valid_values, valid_weights, lower)
                upper_bound = weighted_quantile(valid_values, valid_weights, upper)
            else:
                lower_bound = np.quantile(valid_values, lower)
                upper_bound = np.quantile(valid_values, upper)

            clipped_values = values.astype(float, copy=True)
            clipped_values[valid_mask] = np.clip(valid_values, lower_bound, upper_bound)

            if na_action == 'ignore':
                clipped_values[pd.isna(values)] = np.nan
            elif na_action == 'concerned':
                nan_mask = ~valid_mask
                if nan_mask.any():
                    non_nan_clipped = clipped_values[~nan_mask]
                    if len(non_nan_clipped) > 0:
                        avg_value = np.mean(non_nan_clipped)
                    else:
                        avg_value = 0.0
                    clipped_values[nan_mask] = avg_value

            if (
                pd.api.types.is_integer_dtype(original_dtype)
                and not np.isnan(clipped_values).any()
            ):
                clipped_values = np.round(clipped_values).astype(original_dtype)

            clipped_results[col].loc[group_indices] = clipped_values

    result_df = pd.DataFrame(clipped_results)
    result_df.index = original_index
    return result_df


def promote_historical_ft(file_path, dateid, timeid, lags=MAX_HISTORICAL_TIMESTAMPS):
    """
    获取历史特征排名矩阵
    
    Args:
        file_path: parquet文件路径
        dateid: 目标日期ID
        timeid: 目标时间ID
        lags: 向前读取的时间戳数量
    
    Returns:
        lags × 2 的矩阵，第一列是时序排名，第二列是截面排名
        按时间顺序排列，从最早时间戳到最晚时间戳
    """
    feature_cols = [f'f{i}' for i in range(384)]
    
    timestamps = []
    current_dateid, current_timeid = dateid, timeid
    
    window_size = min(lags, MAX_HISTORICAL_TIMESTAMPS)

    for _ in range(window_size):
        timestamps.append((current_dateid, current_timeid))
        
        if current_dateid == 0 and current_timeid == 0:
            break
        
        current_timeid -= 1
        if current_timeid < 0:
            current_timeid = TIMEIDS_PER_DAY - 1
            current_dateid -= 1
            if current_dateid < 0:
                break
    
    timestamps.reverse()
    
    all_data = []
    for ts_dateid, ts_timeid in timestamps:
        df = read_byclass(file_path, by={'dateid': ts_dateid, 'timeid': ts_timeid})
        if not df.empty:
            all_data.append(df)
    
    if not all_data:
        return np.zeros((window_size, 2))
    
    df = pd.concat(all_data, ignore_index=True)
    
    xsec_ranks = rank(df, columns=feature_cols, by=['dateid', 'timeid'])
    ts_ranks = rank(df, columns=feature_cols, by=['stockid'])
    
    xsec_rank_cols = [f'f{i}_r' for i in range(384)]
    ts_rank_cols = [f'f{i}_r' for i in range(384)]
    
    result_matrix = np.zeros((len(timestamps), 2))
    
    for idx, (ts_dateid, ts_timeid) in enumerate(timestamps):
        mask = (df['dateid'] == ts_dateid) & (df['timeid'] == ts_timeid)
        if mask.any():
            ts_data = df[mask]
            xsec_row = xsec_ranks.loc[ts_data.index, xsec_rank_cols].mean(axis=1).mean()
            ts_row = ts_ranks.loc[ts_data.index, ts_rank_cols].mean(axis=1).mean()
            result_matrix[idx, 0] = ts_row
            result_matrix[idx, 1] = xsec_row
    
    return result_matrix
