import pandas as pd

def read_batch(file_path, batch_size=1000):
    """
    批量读取文件数据
    
    Args:
        file_path: 文件路径
        batch_size: 批次大小
    
    Yields:
        批次数据，类型为 pandas DataFrame
    """
    if file_path.endswith('.parquet'):
        # 读取 parquet 文件
        df = pd.read_parquet(file_path)
        for i in range(0, len(df), batch_size):
            yield df.iloc[i:i+batch_size]
    elif file_path.endswith('.csv'):
        # 读取 csv 文件
        reader = pd.read_csv(file_path, chunksize=batch_size)
        for batch in reader:
            yield batch
    else:
        raise ValueError('Unsupported file format')
