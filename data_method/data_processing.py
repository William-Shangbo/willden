import pandas as pd
import numpy as np


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


def weighted_quantile(values, weights, quantile):
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    cumsum_weights = np.cumsum(sorted_weights)
    cumsum_weights /= cumsum_weights[-1]

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
    
    if isinstance(columns, str):
        columns = [columns]
    
    if isinstance(by, str):
        by = [by]
    
    if isinstance(weights, str):
        weights = [weights]
    
    groups = _construct_groups(df, by)
    
    rank_results = {}
    for col in columns:
        rank_results[f'{col}_r'] = pd.Series(np.full(n_rows, np.nan), index=df.index)
    
    for group_name, group_df in groups:
        if group_df.empty:
            continue
        
        group_indices = group_df.index.values
        
        for col in columns:
            if col not in group_df.columns:
                continue
            
            values = group_df[col].values
            
            if weights:
                weight_values = np.zeros(len(values))
                for w in weights:
                    if w in group_df.columns:
                        weight_values += group_df[w].values
                
                transformed_values = _weight_transform(values, weight_values)
            else:
                transformed_values = values
            
            nan_mask = pd.isna(transformed_values)
            valid_mask = ~nan_mask
            valid_values = transformed_values[valid_mask]
            
            if len(valid_values) == 0:
                continue
            
            sorted_indices = np.argsort(valid_values, kind='stable')
            sorted_values = valid_values[sorted_indices]
            
            positions = np.zeros(len(valid_values))
            
            i = 0
            while i < len(sorted_values):
                j = i
                while j < len(sorted_values) - 1 and sorted_values[j] == sorted_values[j + 1]:
                    j += 1
                
                avg_position = (i + j) / 2.0
                for k in range(i, j + 1):
                    positions[sorted_indices[k]] = avg_position
                
                i = j + 1
            
            if len(positions) > 1:
                normalized_ranks = positions / (len(positions) - 1)
            else:
                normalized_ranks = np.array([0.5])
            
            valid_positions = group_indices[valid_mask]
            for idx, rank_val in zip(valid_positions, normalized_ranks):
                rank_results[f'{col}_r'].loc[idx] = rank_val
    
    result_df = pd.DataFrame(rank_results)
    result_df.index = original_index
    
    return result_df


def standardize(df, columns, by=None, weights=None, na_action='ignore'):
    df = df.copy()
    original_index = df.index.copy()
    
    if isinstance(columns, str):
        columns = [columns]
    
    if isinstance(by, str):
        by = [by]
    
    if isinstance(weights, str):
        weights = [weights]
    
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
            
            values = group_df[col].values
            
            if weights:
                weight_values = np.zeros(len(values))
                for w in weights:
                    if w in group_df.columns:
                        weight_values += group_df[w].values
                
                total_weight = np.sum(weight_values)
                weighted_mean = np.sum(values * weight_values) / total_weight
                
                weighted_variance = np.sum(weight_values * (values - weighted_mean) ** 2) / total_weight
                weighted_std = np.sqrt(weighted_variance)
            else:
                weighted_mean = np.mean(values)
                weighted_std = np.std(values)
            
            if weighted_std == 0:
                zscore = np.full(len(values), np.nan)
            else:
                zscore = (values - weighted_mean) / weighted_std
            
            if na_action == 'ignore':
                zscore[pd.isna(values)] = np.nan
            elif na_action == 'concerned':
                nan_mask = pd.isna(values)
                if nan_mask.any():
                    non_nan_values = values[~nan_mask]
                    if len(non_nan_values) > 0:
                        avg_zscore = np.mean(zscore[~nan_mask])
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
    
    if isinstance(columns, str):
        columns = [columns]
    
    if isinstance(by, str):
        by = [by]
    
    if isinstance(weights, str):
        weights = [weights]
    
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
            values = group_df[col].values
            
            non_nan_mask = ~pd.isna(values)
            non_nan_values = values[non_nan_mask]
            
            if len(non_nan_values) == 0:
                continue
            
            if weights:
                weight_values = np.zeros(len(values))
                for w in weights:
                    if w in group_df.columns:
                        weight_values += group_df[w].values
                
                # 处理权重中的 nan 值，使用与 rank 函数相同的方法
                weight_nan_mask = pd.isna(weight_values)
                non_nan_weights = weight_values[~weight_nan_mask]
                
                if len(non_nan_weights) > 0:
                    avg_weight = np.mean(non_nan_weights)
                else:
                    avg_weight = 1.0
                
                imputed_weights = np.where(weight_nan_mask, avg_weight, weight_values)
                
                lower_bound = weighted_quantile(non_nan_values, imputed_weights[non_nan_mask], lower)
                upper_bound = weighted_quantile(non_nan_values, imputed_weights[non_nan_mask], upper)
            else:
                lower_bound = np.quantile(non_nan_values, lower)
                upper_bound = np.quantile(non_nan_values, upper)
            
            clipped_values = np.clip(values, lower_bound, upper_bound)
            
            if pd.api.types.is_integer_dtype(original_dtype):
                clipped_values = np.round(clipped_values).astype(original_dtype)
            elif pd.api.types.is_float_dtype(original_dtype):
                clipped_values = clipped_values.astype(original_dtype)
            
            if na_action == 'ignore':
                clipped_values[pd.isna(values)] = np.nan
            elif na_action == 'concerned':
                nan_mask = pd.isna(values)
                if nan_mask.any():
                    non_nan_clipped = clipped_values[~nan_mask]
                    if len(non_nan_clipped) > 0:
                        avg_value = np.mean(non_nan_clipped)
                    else:
                        avg_value = 0.0
                    clipped_values[nan_mask] = avg_value
            
            clipped_results[col].loc[group_indices] = clipped_values
    
    result_df = pd.DataFrame(clipped_results)
    result_df.index = original_index
    
    return result_df
