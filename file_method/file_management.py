import pandas as pd
import pyarrow.parquet as pq

# 预定义的主键范围（常量）
STOCKID_RANGE = (0, 499)
DATEID_RANGE = (0, 359)
TIMEID_RANGE = (0, 238)

def read_batch(file_path, batch_size=1000):
    """
    批量读取文件数据，返回固定大小的 DataFrame
    
    Args:
        file_path: 文件路径
        batch_size: 批次大小
    
    Yields:
        批次数据，类型为 pandas DataFrame
    """
    if file_path.endswith('.parquet'):
        # 使用 pyarrow 进行更高效的 parquet 文件读取
        parquet_file = pq.ParquetFile(file_path)
        # 分批次读取，使用 pyarrow 的迭代器
        # 这种方式可以处理非常大的文件
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            yield batch.to_pandas()
    elif file_path.endswith('.csv'):
        # 读取 csv 文件
        reader = pd.read_csv(file_path, chunksize=batch_size)
        for batch in reader:
            yield batch
    else:
        raise ValueError('Unsupported file format')

def read_byclass(file_path, by, batch_size=1000):
    """
    按条件读取文件数据，利用数据的排序特性直接计算行范围
    
    数据排序规则: stockid -> dateid -> timeid
    每个 stockid 预期行数: 360 * 239 = 86040
    
    Args:
        file_path: 文件路径
        by: 过滤条件字典，格式为 {列名: 值或值列表, ...}
        batch_size: 批次大小（此参数保留以保持接口兼容性，当前实现中未使用）
    
    Returns:
        符合条件的所有数据，类型为 pandas DataFrame
    """
    result_list = []
    
    # 获取文件总行数和 row group 信息
    parquet_file = pq.ParquetFile(file_path)
    total_rows = parquet_file.metadata.num_rows
    num_row_groups = parquet_file.metadata.num_row_groups
    
    # 每个 stockid 的预期行数
    rows_per_stockid = 360 * 239
    
    # 计算实际的 stockid 数量
    actual_stockid_count = total_rows // rows_per_stockid
    
    # 确定需要读取的 stockid 范围
    stockid_range = None
    if 'stockid' in by:
        target_stockids = by['stockid']
        if not isinstance(target_stockids, (list, range)):
            target_stockids = [target_stockids]
        stockid_range = (min(target_stockids), max(target_stockids))
    else:
        # 如果没有指定 stockid，需要读取所有 stockid
        stockid_range = (0, actual_stockid_count - 1)
    
    # 计算行范围
    start_row = stockid_range[0] * rows_per_stockid
    end_row = min((stockid_range[1] + 1) * rows_per_stockid, total_rows)
    
    # 计算需要读取的 row groups
    # 每个 row group 大约包含 rows_per_stockid 行
    rows_per_row_group = total_rows // num_row_groups
    
    start_row_group = start_row // rows_per_row_group
    end_row_group = min(end_row // rows_per_row_group + 1, num_row_groups)
    
    # 读取需要的 row groups
    for row_group_idx in range(start_row_group, end_row_group):
        table = parquet_file.read_row_group(row_group_idx)
        batch = table.to_pandas()
        
        # 对批次应用过滤条件
        filtered_batch = batch
        for col, values in by.items():
            if col in filtered_batch.columns:
                if not isinstance(values, (list, range)):
                    values = [values]
                filtered_batch = filtered_batch[filtered_batch[col].isin(values)]
            else:
                filtered_batch = pd.DataFrame()
                break
        
        if not filtered_batch.empty:
            result_list.append(filtered_batch)
    
    # 合并所有结果
    if result_list:
        return pd.concat(result_list, ignore_index=True)
    else:
        return pd.DataFrame()