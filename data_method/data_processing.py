import pandas as pd
import numpy as np

def _weight_transform(values, weights):
    """
    对分组后的 value 和 weight 进行加权变换
    通过相同的计算方法得到归一化后的累积权重（范围 [0, 1]）
    
    Args:
        values: 数值数组
        weights: 权重数组
    
    Returns:
        归一化后的累积权重数组（范围 [0, 1]）
    """
    sorted_indices = np.argsort(values)
    sorted_weights = weights[sorted_indices]
    
    cumsum_weights = np.cumsum(sorted_weights)
    cumsum_weights /= cumsum_weights[-1]
    
    return cumsum_weights

def weighted_quantile(values, weights, quantile):
    """
    计算加权分位数
    
    Args:
        values: 数值数组
        weights: 权重数组
        quantile: 分位数（0-1之间的值）
    
    Returns:
        加权分位数值
    """
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    cumsum_weights = np.cumsum(sorted_weights)
    cumsum_weights /= cumsum_weights[-1]
    
    return np.interp(quantile, cumsum_weights, sorted_values)

def _construct_groups(df, by, valid_mask=None):
    """
    构建分组
    
    Args:
        df: 要处理的 DataFrame
        by: 分组列名
        valid_mask: 有效行的掩码
    
    Returns:
        分组对象
    """
    if valid_mask is not None:
        df = df[valid_mask]
    
    if by:
        groups = df.groupby(by, dropna=False)
    else:
        # 当 by 为 None 时，整个 DataFrame 作为一个组
        class SingleGroup:
            def __iter__(self):
                yield (None, df)
        groups = SingleGroup()
    
    return groups

def rank(df, columns, by=None, weights=None, epsilon=1e-8):
    """
    计算 columns 的排名，精度为 1e-8，na_action 强制设定为 ignore
    排名在 [0, 1] 之间均匀分布
    
    Args:
        df: 要处理的 DataFrame
        columns: 要计算排名的列名，可以是字符串或列表
        by: 分组列名，在每一组内计算排名，如果为 None 则对所有行计算
        weights: 权重列名，在计算排名时使用加权，如果为 None 则直接计算排名
        epsilon: 排名精度，默认 1e-8
    
    Returns:
        包含排名列的 DataFrame，列名为 {原列名}_rank
    """
    df = df.copy()
    
    # 确保 columns 是列表
    if isinstance(columns, str):
        columns = [columns]
    
    # 确保 by 是列表
    if isinstance(by, str):
        by = [by]
    
    # 确保 weights 是列表
    if isinstance(weights, str):
        weights = [weights]
    
    # 强制 na_action 为 ignore
    na_action = 'ignore'
    
    # 处理缺失值
    valid_mask = pd.Series(True, index=df.index)
    if na_action == 'ignore':
        for col in columns:
            valid_mask = valid_mask & ~pd.isna(df[col])
        
        if weights:
            for w in weights:
                valid_mask = valid_mask & ~pd.isna(df[w])
    
    # 构建分组
    groups = _construct_groups(df, by, valid_mask)
    
    # 处理每个组
    result_list = []
    for group_name, group_df in groups:
        if group_df.empty:
            continue
        
        group_result = group_df.copy()
        
        for col in columns:
            if col not in group_result.columns:
                continue
            
            valid_values = group_result[col].values
            
            # 计算排名
            if weights:
                # 加权排名
                weight_values = np.zeros(len(valid_values))
                for w in weights:
                    if w in group_result.columns:
                        weight_values += group_result[w].values
                
                # 使用 _weight_transform 得到归一化后的值
                transformed_values = _weight_transform(valid_values, weight_values)
                
                # 计算排名（基于归一化后的值）
                sorted_indices = np.argsort(transformed_values, kind='stable')
                sorted_transformed = transformed_values[sorted_indices]
                
                # 计算排名
                ranks = np.zeros(len(valid_values))
                for i in range(len(valid_values)):
                    # 找到第一个大于等于当前值的位置
                    rank = np.searchsorted(sorted_transformed, transformed_values[i], side='left') + 1
                    ranks[sorted_indices[i]] = rank
                
                # 归一化到 [0, 1] 范围
                if len(ranks) > 1:
                    normalized_ranks = (ranks - 1) / (len(ranks) - 1)
                else:
                    normalized_ranks = np.array([0.5])  # 只有一个值时，设为 0.5
                
                group_result[f'{col}_rank'] = normalized_ranks
            else:
                # 普通排名
                # 使用 stable 排序确保相同值有相同排名
                sorted_indices = np.argsort(valid_values, kind='stable')
                sorted_values = valid_values[sorted_indices]
                
                # 计算排名
                ranks = np.zeros(len(valid_values))
                for i in range(len(valid_values)):
                    # 找到第一个大于等于当前值的位置
                    rank = np.searchsorted(sorted_values, valid_values[i], side='left') + 1
                    ranks[sorted_indices[i]] = rank
                
                # 归一化到 [0, 1] 范围
                if len(ranks) > 1:
                    normalized_ranks = (ranks - 1) / (len(ranks) - 1)
                else:
                    normalized_ranks = np.array([0.5])  # 只有一个值时，设为 0.5
                
                group_result[f'{col}_rank'] = normalized_ranks
        
        result_list.append(group_result)
    
    # 合并结果
    if result_list:
        result = pd.concat(result_list)
    else:
        result = df.copy()
    
    # 处理无效行
    if valid_mask is not None:
        invalid_df = df[~valid_mask].copy()
        result = pd.concat([result, invalid_df])
    
    return result

def standardize(df, columns, by=None, weights=None, na_action='ignore'):
    """
    计算 zscore，by 决定组别，weights 决定 mean 和 std 的权重
    如果出现 std == 0 的情况，处理为 nan 而不是 inf
    na_action 决定了分类方式
    对每一行生成不同的 zscore 值
    
    Args:
        df: 要处理的 DataFrame
        columns: 要计算 zscore 的列名，可以是字符串或列表
        by: 分组列名，在每一组内计算 zscore，如果为 None 则对所有行计算
        weights: 权重列名，在计算 mean 和 std 时使用加权，如果为 None 则直接计算
        na_action: 处理缺失值的方式
    
    Returns:
        包含 zscore 列的 DataFrame，列名为 {原列名}_zscore
    """
    df = df.copy()
    
    # 确保 columns 是列表
    if isinstance(columns, str):
        columns = [columns]
    
    # 确保 by 是列表
    if isinstance(by, str):
        by = [by]
    
    # 确保 weights 是列表
    if isinstance(weights, str):
        weights = [weights]
    
    # 处理缺失值
    valid_mask = pd.Series(True, index=df.index)
    if na_action == 'ignore':
        for col in columns:
            valid_mask = valid_mask & ~pd.isna(df[col])
        
        if weights:
            for w in weights:
                valid_mask = valid_mask & ~pd.isna(df[w])
    
    # 构建分组
    groups = _construct_groups(df, by, valid_mask)
    
    # 处理每个组
    result_list = []
    for group_name, group_df in groups:
        if group_df.empty:
            continue
        
        group_result = group_df.copy()
        
        for col in columns:
            if col not in group_result.columns:
                continue
            
            valid_values = group_result[col].values
            
            # 计算加权 mean 和 std
            if weights:
                # 加权 mean 和 std
                weight_values = np.zeros(len(valid_values))
                for w in weights:
                    if w in group_result.columns:
                        weight_values += group_result[w].values
                
                total_weight = np.sum(weight_values)
                weighted_mean = np.sum(valid_values * weight_values) / total_weight
                
                # 加权标准差
                weighted_variance = np.sum(weight_values * (valid_values - weighted_mean) ** 2) / total_weight
                weighted_std = np.sqrt(weighted_variance)
            else:
                # 普通 mean 和 std
                weighted_mean = np.mean(valid_values)
                weighted_std = np.std(valid_values)
            
            # 处理 std == 0 的情况
            if weighted_std == 0:
                zscore = np.full(len(valid_values), np.nan)
            else:
                zscore = (valid_values - weighted_mean) / weighted_std
            
            group_result[f'{col}_zscore'] = zscore
        
        result_list.append(group_result)
    
    # 合并结果
    if result_list:
        result = pd.concat(result_list)
    else:
        result = df.copy()
    
    # 处理无效行
    if valid_mask is not None:
        invalid_df = df[~valid_mask].copy()
        result = pd.concat([result, invalid_df])
    
    return result

def winsorize(df, columns, by=None, weights=None, lower=0.05, upper=0.95, na_action='ignore'):
    """
    对指定列进行 winsorize 处理（掐头去尾）
    
    Args:
        df: 要处理的 DataFrame
        columns: 要进行 winsorize 处理的列名，可以是字符串或列表
        by: 分组列名，在每一组内进行 winsorize 处理，如果为 None 则对所有行处理
        weights: 权重列名，在计算分位数时使用加权平均，如果为 None 则直接计算分位数
        lower: 下分位数界，默认 0.05
        upper: 上分位数界，默认 0.95
        na_action: 处理缺失值的方式
            - 'ignore': 如果任何使用到的变量（columns, by, 或 weights）为缺失值，忽略这一行
            - 'interested': 当 columns=nan 时，fillna(0)；当 by=nan 时，单独分出一组；当 weights=nan 时，使用组内均值
    
    Returns:
        处理后的 DataFrame，支持 df[target_cols] = winsorize(df, ...) 的用法
    """
    df = df.copy()
    
    # 确保 columns 是列表
    if isinstance(columns, str):
        columns = [columns]
    
    # 确保 by 是列表
    if isinstance(by, str):
        by = [by]
    
    # 确保 weights 是列表
    if isinstance(weights, str):
        weights = [weights]
    
    # 处理缺失值
    valid_mask = pd.Series(True, index=df.index)
    if na_action == 'ignore':
        for col in columns:
            valid_mask = valid_mask & ~pd.isna(df[col])
        
        if weights:
            for w in weights:
                valid_mask = valid_mask & ~pd.isna(df[w])
    
    # 构建分组
    groups = _construct_groups(df, by, valid_mask)
    
    # 处理每个组
    result_list = []
    for group_name, group_df in groups:
        if group_df.empty:
            continue
        
        group_result = group_df.copy()
        
        for col in columns:
            if col not in group_result.columns:
                continue
            
            # 保存原始数据类型
            original_dtype = group_result[col].dtype
                
            valid_values = group_result[col].values
            
            # 计算分位数
            if weights:
                # 加权分位数
                weight_values = np.zeros(len(valid_values))
                for w in weights:
                    if w in group_result.columns:
                        weight_values += group_result[w].values
                
                lower_bound = weighted_quantile(valid_values, weight_values, lower)
                upper_bound = weighted_quantile(valid_values, weight_values, upper)
            else:
                # 普通分位数
                lower_bound = np.quantile(valid_values, lower)
                upper_bound = np.quantile(valid_values, upper)
            
            # 应用 winsorize
            clipped_values = np.clip(
                group_result[col].values, 
                lower_bound, 
                upper_bound
            )
            
            # 转换回原始数据类型
            if pd.api.types.is_integer_dtype(original_dtype):
                clipped_values = np.round(clipped_values).astype(original_dtype)
            elif pd.api.types.is_float_dtype(original_dtype):
                clipped_values = clipped_values.astype(original_dtype)
            
            group_result[col] = clipped_values
        
        result_list.append(group_result)
    
    # 合并结果
    if result_list:
        result = pd.concat(result_list)
    else:
        result = df.copy()
    
    # 处理无效行
    if valid_mask is not None:
        invalid_df = df[~valid_mask].copy()
        result = pd.concat([result, invalid_df])
    
    return result
