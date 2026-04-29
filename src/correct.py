import pandas as pd
import numpy as np
import ast

# 读取CSV文件
file_path = 'mol_rep.csv'  # 替换为你的文件路径
df = pd.read_csv(file_path)

# 定义转换函数
def convert_to_target_format(molecular_rep):
    try:
        # 将字符串转换为列表
        array = np.array(ast.literal_eval(molecular_rep.strip()))
        # 将数组转换为目标格式
        return list(array.astype(float))
    except Exception as e:
        print(f"Error processing: {molecular_rep}, Error: {e}")
        return None

# 应用转换函数到'molecularRep'列
df['molecularRep'] = df['molecularRep'].apply(convert_to_target_format)

# 保存转换后的文件
output_file_path = 'mol_rep_14.csv'
df.to_csv(output_file_path, index=False)

print(f"File saved to {output_file_path}")
