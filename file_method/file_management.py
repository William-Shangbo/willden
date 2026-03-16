import pandas as pd
import pyarrow.parquet as pq

def read_batch(file_path, batch_size=1000, by=None):
    """
    批量读取文件数据
    
    Args:
        file_path: 文件路径
        batch_size: 批次大小
        by: 按列分组的列名（可选）
    
    Yields:
        批次数据，类型为 pandas DataFrame
    """
    if file_path.endswith('.parquet'):
        # 使用 pyarrow 进行更高效的 parquet 文件读取
        parquet_file = pq.ParquetFile(file_path)
        
        if by:
            # 先获取所有唯一的分组值，避免一次性加载全部数据
            unique_values = set()
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                batch_df = batch.to_pandas()
                if by in batch_df.columns:
                    unique_values.update(batch_df[by].unique())
            
            # 对每个唯一值，分批读取数据并过滤
            for value in unique_values:
                # 分批读取并过滤，避免一次性加载全部数据
                for batch in parquet_file.iter_batches(batch_size=batch_size):
                    batch_df = batch.to_pandas()
                    if by in batch_df.columns:
                        filtered_batch = batch_df[batch_df[by] == value]
                        if not filtered_batch.empty:
                            yield filtered_batch
        else:
            # 分批次读取，使用 pyarrow 的迭代器
            # 这种方式可以处理非常大的文件
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                yield batch.to_pandas()
    elif file_path.endswith('.csv'):
        # 读取 csv 文件
        reader = pd.read_csv(file_path, chunksize=batch_size)
        for batch in reader:
            if by and by in batch.columns:
                # 按列分组读取
                for value in batch[by].unique():
                    group_df = batch[batch[by] == value]
                    yield group_df
            else:
                yield batch
    else:
        raise ValueError('Unsupported file format')
