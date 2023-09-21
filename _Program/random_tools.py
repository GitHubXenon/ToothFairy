import numpy as np
import utility_functions as uf
import filter as f
from scipy.io import wavfile

# 随机方案
GAUSS = 1
MEAN = -1


# 【高斯随机调整器】
# data：待处理数据，可以为数组或标量，标量仅支持 python 原生 int 型和 float 型
# loc：均值，或称正态分布的中心位置
# scale：标准差，或称正态分布的随机范围
# float_range：表示随机幅度，小于 0 表示默认，单值为 1/20 的浮动范围，数组为最大减最小。
# rounding：表示结果是否取整
# variance 越大，表示离散程度越大
def gauss_rand(data, mean=0., variance=1., float_range=-1, around=False):
    # 常规数据校验
    shape = None
    data_type = uf.NUM
    if uf.is_num(data):
        shape = None
        data_type = uf.NUM
    elif uf.is_ary(data):
        data = np.array(data, dtype=float)
        shape = data.shape
        data_type = uf.ARY
    else:
        print('输入数据不合法！')
        return None
    # 最后一个参数 shape 可以省略，省略时获得的是单个数值。
    gauss_random_array = np.random.normal(mean, variance, shape)
    if float_range < 0:
        if data_type == uf.NUM:
            # 计算数组最大值和最小值，求差获得随机浮动范围
            float_range = np.abs(data / 20)
        elif data_type == uf.ARY:
            float_range = np.max(data) - np.min(data)
    data = data + gauss_random_array * float_range
    # 这里不能使用 += 语法，因为数据类型可能会报错。
    if around:
        data = int(np.rint(data))
    return data


# 【均分随机调整器】
# 随机偏移量的区间，一般是正负对称的形式，例如 min_val = -10, max_val = 10
def mean_rand(data, min_val, max_val, around=False):
    # 常规数据校验
    shape = None
    data_type = uf.NUM
    if uf.is_num(data):
        shape = None
        data_type = uf.NUM
    elif uf.is_ary(data):
        data = np.array(data, dtype=float)
        shape = data.shape
        data_type = uf.ARY
    else:
        print('输入数据不合法！')
        return None
    mean_random_array = np.random.uniform(min_val, max_val, shape)
    data = data + mean_random_array
    # 这里不能使用 += 语法，因为数据类型可能会报错。
    if around:
        data = int(np.rint(data))
    return data


# 【混合随机调整器】
# 按照一定比例使用均分随机和高斯随机，输入的 data 必须为数组
# mean_prop 表示均分随机的概率
# outlier 表示出现异常值的概率，异常值为 1-2 倍随机幅度？
def mixed_rand(data,
               min_val=np.NaN, max_val=np.NaN,
               mean_prop=0.5, mean=0., variance=1., float_range=-1,
               outlier_prop=0.1, around=False):
    # 数据校验
    shape = None
    data_type = uf.NUM
    if uf.is_num(data):
        print('不能输入单个数值！')
        return None
    elif uf.is_ary(data):
        data = np.array(data, dtype=float)
        shape = data.shape
        data_type = uf.ARY
    else:
        print('输入数据不合法！')
        return None

    # 随机幅度
    if float_range < 0:
        if data_type == uf.NUM:
            # 计算数组最大值和最小值，求差获得随机浮动范围
            float_range = np.abs(data / 20)
        elif data_type == uf.ARY:
            float_range = np.mean(data) / 10
    if np.isnan(min_val):
        min_val = -float_range / 2
    if np.isnan(max_val):
        max_val = float_range / 2

    # 随机方法的数组，数组取值为 [0, 1]，大于概率值则执行 gauss_rand，否则执行 mean_rand
    rand_method_prob = np.random.random((len(data),))
    outlier_prob_ary = np.random.random((len(data),))
    for i in range(0, len(data)):
        if rand_method_prob[i] >= mean_prop:
            if outlier_prob_ary[i] < outlier_prop:
                # 异常值
                data[i] = gauss_rand(data[i], mean, variance, (np.random.random() + 5) * float_range, around)
            else:
                data[i] = gauss_rand(data[i], mean, variance, float_range, around)
        else:
            if outlier_prob_ary[i] < outlier_prop:
                # 异常值
                data[i] = mean_rand(data[i],
                                    (np.random.random() + 1) * min_val,
                                    (np.random.random() + 1) * max_val,
                                    around)
            else:
                data[i] = mean_rand(data[i], min_val, max_val, around)
    return data


# 【真值概率随机】
# prop 表示真值概率
# 返回 True 表示出现了特殊值，False 表示未出现特殊值
# 多种概率随机就可以使用 if 语句进行复合
def prop_rand(prop=0.5):
    if not 0 <= prop <= 1:
        print("prop_rand 错误：概率值", prop, "不正确。")
        return False
    if np.random.random() < prop:
        return True
    else:
        return False


# 从一个集合中随机抓取几个元素
def catch_balls(data, num):
    return


"""
从集合中抽取若干元素（不重复）
from random import sample
l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(sample(l, 5)) # 随机抽取5个元素
'''
[5, 10, 1, 4, 2]
'''

取整数：


"""


# 随机位置删除
# num 表示删除个数
def rand_delete(data, num):
    # replace 表示是否可以被多次选择
    # data.shape[0] = len(data)
    delete_index = np.random.choice(len(data), num, replace=False)
    data = np.delete(data, delete_index)
    return data


# 获取随机位置
def get_rand_pos(data, num):
    return np.random.choice(len(data), num, replace=False)


# 取整数随机数
def get_randint(low, high):
    return np.random.randint(low, high)


# 注：正态分布的三个主要范围：± 1, 1.96, 2.58

# 【获取随机位置的截选序列】从真实信号中获取数据
def get_rand_len_signal(f_path, start=0, end=0, length=1.5e5, float_range=2e4):
    fs, signal = wavfile.read(f_path)
    length_ = gauss_rand(length, around=True, float_range=int(float_range))
    # print(length)
    if end == 0:
        end = len(signal)
    # 这里 end-length 可能 <= 0，所以这里使其初始化为 len，才能进入以下循环
    idx = len(signal)
    while (idx + length) >= len(signal):
        # 防止溢出
        # length = mixed_rand(length, around=True, min_val=-int(float_range / 2), max_val=int(float_range / 2),
        #                     float_range=int(float_range))
        length_ = gauss_rand(length, float_range=int(float_range), around=True)
        idx = np.random.randint(start, end - length)
    print("信号切割位置：[", idx, ':', idx + length_, '] /', len(signal))
    return fs, signal[idx:idx + length_]


# 【信号随机器】将一个信号生成随机的、大体相同但细节不同的信号。
# 该算法专用于声音信号，暂不能用于其他领域。
def get_rand_signal(data, rounding=True):
    data = np.array(data, dtype=float)
    end_idx = f.get_end_idx(data)
    # 记录原始端点值
    raw_end = np.array([], dtype=float)
    for i in range(0, len(end_idx)):
        raw_end = np.append(raw_end, data[end_idx[i]])
    mean_interval = np.mean(np.diff(end_idx)) * 2
    # 找到 sigmoid 的最佳参数
    offset, width = uf.get_sigmoid_param([0, mean_interval])
    # 端点值调整
    for i in range(0, len(end_idx) - 1):
        # 最后一个点就不调整了
        float_range = uf.sigmoid(end_idx[i + 1] - end_idx[i], offset, width) * 10
        data[end_idx[i]] = gauss_rand(data[end_idx[i]],
                                      mean=0,
                                      variance=1,
                                      float_range=float_range)
    # 这里分两个循环，因为所有端点值都调整完后，才能调整区间值。
    # 区间调整为双层循环，raw_end 的最后一个值与对应位置新值相比没有变化。
    for i in range(0, len(end_idx) - 1):
        interval_len = end_idx[i + 1] - end_idx[i]
        # 区间左右端点的调整倍数
        left_mpl = data[end_idx[i]] / raw_end[i]
        right_mpl = data[end_idx[i + 1]] / raw_end[i + 1]
        for j in range(1, interval_len):
            mpl = (right_mpl - left_mpl) / interval_len * j + left_mpl
            data[end_idx[i] + j] = data[end_idx[i] + j] * mpl
    if rounding:
        data = np.int32(np.rint(data))
    return data


"""
该随机方案有个问题:
容易出现各种频率成分复杂的信号
但是为了获取仿真的随机信号又必须对所有模糊峰值点进行大的调整

那么就要提出弥补方案:
调整前,将频域的一阶差分大于一定值的(也就是陡峰)的频率记录下来
记录频谱的陡峰的峰值
然后计算非峰值部分的功率
然后滤波
滤波后重新计算非峰值功率并进行调整,对整体功率进行调整,使非峰值与原始非峰值功率一致
然后
然后再向原峰值位置重新添加新的cos信号,信号的振幅是原始与新的之差

这个方案问题在于,峰值也有宽有窄,怎么衡量峰值的宽窄?这是一个一直都困扰的问题
方案:使用一阶差分来计算,通过计算变号点左右 abs 值大于一定范围


窄峰和宽峰怎么调整的问题?
峰值一定是左正右负
遇到左负右正的为谷,也就是峰的结束位置?
通过峰值左右的长度来定义


"""

"""
随机调整方案：
    1. 计算时域差分
    2. 通过遍历，找到差分中符号变化的两条线，即 data_diff[i-1]*data_diff[i]<0（若为 0 的地方忽略）
        连线的逻辑是：线[i] 连接着点[i] 和点[i+1]
        那么当线 [i-1] 和 [i] 满足条件时，断点应为 data[i]，即为时域的峰或谷
        len(线) = len(点)-1
        那么应该 i in range(1, len(线)) 
        遍历成功的下标存入数组 end_idx
    3. 将 end_idx 首尾各加入新的数据 0 和 len(data)-1
    4. 上一个循环结束后，进行二次循环，该循环有两个任务：
        a. 保存原始的端点值数组 raw_end，该数组长度应该与 end_idx 长度相同
        b. 将端点进行随机偏移处理：随机程度是百分之一的平均振幅
            data[end_idx[i]] = gauss_rand(data[end_idx[i]], float_range=float_range)
    5. 进行三次循环，遍历所有端点间隔的区间：
        a. 遍历方法是：i in range(1, len(end_idx))
            其区间数据为
            data[end_idx[i-1]:end_idx[i]]
        b. 操作是：
            1) 计算 multiple = np.around((data[end_idx[i]]-data[end_idx[i-1]])/(raw[i]-raw[i-1]),6)
                这里会不会有除法溢出？
            2) 区间伸缩 data[end_idx[i-1]:end_idx[i]] = data[end_idx[i-1]:end_idx[i]]*multiple

为避免出现杂乱的高频分量，随机幅度还要根据区间长度自适应。
这个函数应该是 x 越接近 0，则 y 越快接近 0
x 取值较小时，y 值接近。
x 取值较大时，也能保证不超过一定的值。
"""

"""
生成高斯白噪声的代码：
序列为 X，长度为 N，信噪比为 snr
https://blog.csdn.net/QKK612501/article/details/115953776

import numy as np

def generate_white_noise(X, N, snr)
	noise = np.random.randn(N)
	snr = 10 ** (snr/10)
	power = np.mean(np.square(X))
	npower = power / snr
	noise = noise * np.sqrt(npower)

实际上也就是直接将高斯随机数组添加进去
"""
