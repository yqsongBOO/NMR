# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 读取CSV文件
# df = pd.read_csv('mean_distribution.csv')  # 请替换为你的CSV文件路径
#
# # 获取'值'列
# x = df['Value']
#
# # 计算y值
# y = np.exp(-0.5 * ((x - 5) / 0.1) ** 2)
#
# # 绘制散点图
# plt.scatter(x, y, s=1)
# plt.xlabel('Value')
# plt.ylabel('y值')
# plt.title('散点图：y = exp(-0.5 * ((x - 5) / 0.1)^2)')
# plt.show()

import plotly.express as px
import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('mean_distribution.csv')  # 替换为你的文件路径

# 获取'值'列
x = df['Value']

# 计算 y 值
y = np.exp(-0.5 * ((x - 5) / 0.1) ** 2)

# 创建 DataFrame
data = pd.DataFrame({'Value': x, 'Y值': y})

# 使用 Plotly 绘制交互式散点图
fig = px.scatter(data, x='Value', y='Y值', title='交互式散点图')

# 显示图表
fig.show()




# data = pd.read_csv('mean_distribution.csv') # 示例数据
#
# # 使用 Plotly 绘制直方图
# fig = px.histogram(data, x='Value', y='Count', nbins=1000, title='值的分布直方图', labels={'Value': '值'})
# y_max = data['Count'].max()
# # 设置中文标题和标签
# fig.update_layout(
#     xaxis_title='值',
#     yaxis_title='频数',
#     title_font=dict(family='Arial', size=16),  # 设置标题字体大小
#     xaxis=dict(tickfont=dict(size=12)),       # 设置 x 轴刻度字体大小
#     yaxis=dict(tickfont=dict(size=12),range=[0, y_max*1]),
#     margin=dict(l=50, r=50, t=50, b=50) # 设置 y 轴刻度字体大小
# )
#
# # 显示图形
# fig.show()




