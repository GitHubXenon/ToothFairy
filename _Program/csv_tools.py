import pandas as pd
import numpy as np

# 该模块用于读写 CSV 文件


"""
pandas 的基础数据类型是 DataFrame，简称 df
一般从 numpy 写入 csv 不用先转换为 df 再写入的方法，因为速度奇慢

CSV 数据的存储结构：换行符区分不同行，逗号区分不同列，可以用记事本打开

在 pandas 中，列有列名，行没有名，这种数据组织方式类似于数据库
取列有两种方式：按名取和按索引取
按名取 df['a']
按索引取 df.iloc[:,0] 未知列名，iloc 意思是 index location

获取行列数
df.shape #返回df的行数和列数
df.shape[0] #返回df的行数
df.shape[1] #返回df的列数
"""


# 将 csv 文件读入为 mat
# 行为数据，列为广义表
def read_csv(path):
    df = pd.read_csv(path)
    # 读入直接转置即可
    return np.array(df).T


# 将 csv 文件写入
def write_csv(data, path, col_names=()):
    # 对每一列数据加列名
    if len(col_names) != len(data):
        # 若列名未指定，就从 0 开始编号
        col_names = np.arange(0, len(data))
    for i in range(len(data)):
        data[i] = np.insert(data[i], 0, col_names[i])
    data = np.array(data).T
    # 这里需要转置一下，因为 csv 存储以列为广义表

    # 使用 tofile 的方法会写入为一维数组（自动执行 flatten 操作）
    # data.T.tofile(path, sep=',', format='%f')
    np.savetxt(path, data, delimiter=',', fmt='%f')
    return


# 以下为代码示例
# csv_data = pd.read_csv(r"C:\Users\xenon\Desktop\包络图.csv")
# 类型是 DataFrame
# print(type(csv_data))
# print(csv_data)
# 怎么取数据？用列名取一整列，再用索引取单个数据，可以用 uf.list2str()
# print(csv_data["x"][0])
# print(uf.list2str(csv_data["x"]))
# print(uf.list2str(csv_data["Curve1"]))

"""
Engauge Digitizer 用法
1. 标三个坐标轴点，确定坐标轴。
2. 线段填充工具（绿色）
3. 文件 - 导出

设置识别分辨率（线段点数）
设置 - 分段填充 - 点分离
"""
