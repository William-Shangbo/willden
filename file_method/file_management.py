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
            # 按列分组读取（需要先获取所有数据，对于大文件可能会占用较多内存）
            # 这种方式对于非常大的文件可能不适用
            df = parquet_file.read().to_pandas()
            for value in df[by].unique():
                group_df = df[df[by] == value]
                for i in range(0, len(group_df), batch_size):
                    yield group_df.iloc[i:i+batch_size]
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
