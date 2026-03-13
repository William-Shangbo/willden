# willden
fundamental facilities for data engineering, such as statistical analysis, feature digging, and plotting

## 安装

```bash
# 从本地安装
pip install -e .
```

## 数据处理函数

### 1. rank 函数

计算指定列的排名，排名在 [0, 1] 之间均匀分布。

**参数：**
- `df`: 要处理的 DataFrame
- `columns`: 要计算排名的列名，可以是字符串或列表
- `by`: 分组列名，在每一组内计算排名，如果为 None 则对所有行计算
- `weights`: 权重列名，在计算排名时使用加权，如果为 None 则直接计算排名
- `epsilon`: 排名精度，默认 1e-8

**返回值：**
包含排名列的 DataFrame，列名为 {原列名}_rank

**示例：**
```python
from willden.data_method.data_processing import rank

# 计算加权排名
df['value_rank'] = rank(df, columns='value', by='group', weights='weight')
```

### 2. standardize 函数

计算指定列的 z-score（标准化值）。

**参数：**
- `df`: 要处理的 DataFrame
- `columns`: 要计算 z-score 的列名，可以是字符串或列表
- `by`: 分组列名，在每一组内计算 z-score，如果为 None 则对所有行计算
- `weights`: 权重列名，在计算 mean 和 std 时使用加权，如果为 None 则直接计算
- `na_action`: 处理缺失值的方式，默认 'ignore'

**返回值：**
包含 z-score 列的 DataFrame，列名为 {原列名}_zscore

**示例：**
```python
from willden.data_method.data_processing import standardize

# 计算加权 z-score
df['value_zscore'] = standardize(df, columns='value', by='group', weights='weight')
```

### 3. winsorize 函数

对指定列进行 winsorize 处理（掐头去尾）。

**参数：**
- `df`: 要处理的 DataFrame
- `columns`: 要进行 winsorize 处理的列名，可以是字符串或列表
- `by`: 分组列名，在每一组内进行 winsorize 处理，如果为 None 则对所有行处理
- `weights`: 权重列名，在计算分位数时使用加权平均，如果为 None 则直接计算分位数
- `lower`: 下分位数界，默认 0.05
- `upper`: 上分位数界，默认 0.95
- `na_action`: 处理缺失值的方式，默认 'ignore'

**返回值：**
处理后的 DataFrame，支持 df[target_cols] = winsorize(df, ...) 的用法

**示例：**
```python
from willden.data_method.data_processing import winsorize

# 对值进行 winsorize 处理
df['value'] = winsorize(df, columns='value', by='group', weights='weight')
```

### 4. weighted_quantile 函数

计算加权分位数。

**参数：**
- `values`: 数值数组
- `weights`: 权重数组
- `quantile`: 分位数（0-1之间的值）

**返回值：**
加权分位数值

**示例：**
```python
from willden.data_method.data_processing import weighted_quantile

# 计算加权中位数
median = weighted_quantile(values, weights, 0.5)
```

### 5. _weight_transform 函数

对分组后的 value 和 weight 进行加权变换，得到归一化后的累积权重。

**参数：**
- `values`: 数值数组
- `weights`: 权重数组

**返回值：**
归一化后的累积权重数组（范围 [0, 1]）

### 6. _construct_groups 函数

构建分组。

**参数：**
- `df`: 要处理的 DataFrame
- `by`: 分组列名
- `valid_mask`: 有效行的掩码

**返回值：**
分组对象

## 性能测试

在 100 万条数据下的执行时间：

| 函数 | 执行时间（秒） | 内存使用（MB） |
|------|---------------|---------------|
| rank | 1.23 | 15.38 |
| winsorize | 0.20 | 7.75 |
| standardize | 0.08 | 15.38 |

## 示例代码

```python
import pandas as pd
import numpy as np
from willden.data_method.data_processing import rank, standardize, winsorize

# 创建测试数据
df = pd.DataFrame({
    'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'group': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B', 'C', 'A'],
    'weight': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.5, 0.6, 0.7, 0.8]
})

# 计算排名
df = rank(df, columns='value', by='group', weights='weight')

# 计算 z-score
df = standardize(df, columns='value', by='group', weights='weight')

# 进行 winsorize 处理
df['value_winsorized'] = winsorize(df, columns='value', by='group', weights='weight')

print(df)
```
