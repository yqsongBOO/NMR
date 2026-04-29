import pandas as pd

# 文件路径
file_path = "/public/home/ustc_yangqs/20/CHnmr/CHnmr_pyg/raw/tokenized_dataset_N_new.csv"

try:
    # 使用pandas读取CSV文件
    # 注意：对于大文件，可以使用chunksize参数分块读取
    df = pd.read_csv(file_path)

    # 获取数据量（行数）
    data_count = len(df)

    print(f"文件 '{file_path}' 中的数据量为: {data_count} 行")

except FileNotFoundError:
    print(f"错误: 文件 '{file_path}' 未找到")
except Exception as e:
    print(f"处理文件时发生错误: {str(e)}")