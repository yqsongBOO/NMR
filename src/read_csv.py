import pandas as pd

# 读取原始 CSV 文件
input_file = '/public/home/ustc_yangqs/molecular2molecular/src/mol_rep.csv'  # 输入文件路径
output_file = '/public/home/ustc_yangqs/molecular2molecular/src/mol_rep_5.csv'  # 输出文件路径

# 使用 pandas 读取 CSV 文件
df = pd.read_csv(input_file)

# 获取前五条记录
df_head = df.head(5)

# 保存前五条记录到新的 CSV 文件
df_head.to_csv(output_file, index=False)

print(f"前五条记录已保存到 {output_file}")
