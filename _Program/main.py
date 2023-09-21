# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np

import csv_tools as ct
import random_tools as rt
import power as p
import view as v
import filter as f
import data_split as ds

from tooth_research import Teeth
import tooth_research as tr
# from sklearn.metrics import confusion_matrix
import threading
import time

"""
6.27 要写的算法：
共振峰判别法的方案：（算法已经想出来了，现在需要实现）
    考虑到振幅不同，需要实现一种对于振幅自适应的方案
        归一化幅频积 = 振幅 × 频率（振幅频率积）/ 振幅均值
    还是减振幅均值？
    这一步的目的是归一化
    振幅乘频率是为了实现加权
    
6.30 改进：
    现在已经获得了对勾函数，利用对勾函数构造真实数据。
    已知的对勾函数过 (40, 500)，那么根据 50Hz 的峰值进行自适应
        通过 get_nearest_idx 和 get_peaks 函数获取 50Hz 的峰值
        找到倍数关系 multiple = 真 / 峰
            推导过程：
                真 / 峰 = multiple，那么实际希望是 0.8 就应该是
                真 / m = 峰
                0.8 * 真 / m = 0.8 * 峰
                实际倍数就是 0.8 / m
        上牙区套用 a = 5，b = 4000，实际倍数 0.5 来算
        下牙区套用 a = 10，b = 4000，实际倍数 0.8 来算
    确定好不同的对勾函数后，按照 1Hz 一个采样加入原始信号，采样的位置从 10Hz - 45Hz
    
数据调整改进：
    浮动范围内，一半是正态，另一半是均分
    
怎么做不同人的数据？
    微调 cd 值，而 ab 值不变
"""


class GenDataThread(threading.Thread):
    def __init__(self, uname, area, depth, num,
                 src_path=r"C:\Users\xenon\OneDrive\_真实数据\【左上】\L4.wav",
                 gen_path=r'F:\python生成的数据',
                 conf_dg=0.05, err_prob=0.1):
        threading.Thread.__init__(self)
        self.uname = uname
        self.area = area
        self.depth = depth
        self.num = num
        self.src_path = src_path
        self.gen_path = gen_path
        self.conf_dg = conf_dg
        self.err_prob = err_prob

    def run(self):
        start_time = time.time()
        for i in range(0, self.num):
            # 设置一定的异常值
            err = np.random.random()
            if err < self.err_prob:
                err = np.random.uniform(1, 3)
                succeed = tr.gen_random_signal_2(self.uname, self.area, self.depth, i,
                                                 self.src_path, self.gen_path, self.conf_dg * err, rand=rt.GAUSS)
            else:
                if i % 2 != 0:
                    # 信号一半采用高斯分布，一半采用平均分布
                    succeed = tr.gen_random_signal_2(self.uname, self.area, self.depth, i,
                                                     self.src_path, self.gen_path, self.conf_dg, rand=rt.GAUSS)
                else:
                    succeed = tr.gen_random_signal_2(self.uname, self.area, self.depth, i,
                                                     self.src_path, self.gen_path, self.conf_dg, rand=rt.MEAN)
            if not succeed:
                print("生成失败")
        stop_time = time.time()
        print('线程结束，用时：', round(stop_time - start_time), 'sec')
        return


# 使用多线程优化生成数据
def gen_data(uname, area, start_depth, stop_depth, num,
             src_path=r"C:\Users\xenon\OneDrive\_真实数据\【左上】\L4.wav",
             gen_path=r'F:\python生成的数据',
             # src_path=r"D:\OneDrive\_真实数据\【左上】\L4.wav",
             # gen_path=r'D:\python生成的数据',
             conf_dg=0.2, err_prob=0.1):
    threads = []
    for i in range(start_depth, stop_depth):
        threads.append(GenDataThread(uname, area, i, num, src_path, gen_path, conf_dg, err_prob))
        threads[-1].start()
    return


"""
坐标轴截断画法
bax = brokenaxes(xlims=((0, .1), (.4, .7)), ylims=((-1, .7), (.79, 1)), hspace=.05, despine=False)
bax.plot

https://www.zhihu.com/column/p/205263612
断轴画法：
画断轴的本质是画两个图，两个图均是同一信号的局部。
将上下局部拼到一起，中间加个断裂带即可。
"""

"""
8 月 17 日晚结果：
alpha 就应该是个正的，但不是定值。拟合出来是 0.25 左右？
因此要拟合三个参数：n、d_max、alpha
其中 alpha 必然是正的，因为 x 与 r 是负相关。
"""

"""
振幅函数的定义规则：
x 取值一定是除以 fs 后，才是真实振幅。
例如：x = 44100，在振幅函数中表示 1。
那么在算法中需要知道某个点的振幅，就需要索引加上采样率。
换句话说，sin_func 获取的横轴单位是秒，振幅函数对应的横轴单位要以秒为单位进行设计。
那么还需要的算法包括：
1. 两个值之间的过度函数，包括直线型、下凹线，上凸线等。
2. 随机波动曲线，给定函数和取值区间，在某个范围内较为平滑地随机波动。（初步考虑多项式拟合法，并打几个随机点）
"""


def gen_data_single(name='LZH', area=Teeth.DL, start_num=10, data_num=3, src_path=r"C:\Users\xenon\Desktop\清洁3~1.wav"):
    start_time = time.time()
    # 生成数据可以使用多线程优化
    for i in range(1, 9):
        for j in range(start_num, start_num + data_num):
            if rt.prop_rand(0.3):
                tr.gen_random_signal_2(name, area, i, j,
                                       src_path=src_path,
                                       gen_path=r"C:\Users\xenon\Desktop\gen",
                                       conf_dg=0.1 * rt.get_randint(3, 9),
                                       rand=rt.GAUSS)
            else:
                if j % 2 == 0:
                    tr.gen_random_signal_2(name, area, i, j,
                                           src_path=src_path,
                                           gen_path=r"C:\Users\xenon\Desktop\gen",
                                           conf_dg=0.1,
                                           rand=rt.GAUSS)
                else:
                    tr.gen_random_signal_2(name, area, i, j,
                                           src_path=src_path,
                                           gen_path=r"C:\Users\xenon\Desktop\gen",
                                           conf_dg=0.1,
                                           rand=rt.MEAN)
    # 实测 0.06 左右就比较逼真
    stop_time = time.time()
    print('用时：', round(stop_time - start_time))
    return


"""
重要结论：
函数参数为 list 或 dict 时，对其进行改写，会修改原数据！
即使用 append 嵌套列表，也会修改原数据。
"""

# ------------------------------
# 主文件入口
if __name__ == '__main__':
    ds.cut(r"C:\Users\xenon\Desktop\数据20230621\新建文件夹\data2", "WY", 120)

    """
    后槽牙：
    组3 2500 5500
    组4 2400 ？
    组5 2700 39000
    
    先回去确定不同人的模型
    
    共振是谐波峰造成的？谐波峰占主要，后面带小峰为次要
    
    当前任务：
    找合适的原始信号用于切割。
    信号来源：外侧面上侧面，上下区分1，上下区分2
    
    20 号以后新增算法：
    滑动窗口查找自然频率计算区间
    
    隐马尔可夫 HMM 怎么构建？
    首先为什么要用HMM，因为真实的刷牙情况中存在连续滑动，且每颗牙之间的转移的关联性是有规律的。
    这里有一个难点，常规的HMM需要通过大量的序列数据和BW算法，统计出转移矩阵
    在实际应用中，用户难以获得 Ground Truth。
    我们发现了一个事实：连续滑动时，仅有三种状态转移结果，即前、自、后
    将模型简化为中间最大，两边次之，接近于正态分布。
    怎么统计这个规律？通过统计刷牙时长和转移时长之差（实验室条件）
    可以校正轨迹数据

23日改：
任务1：绘制特征频带的全局分布图
1. 频带功率的多线程处理，对信号功率计算为多线程，设置二重循环为单线程。
    将所有数据，按照信号个数划分多线程任务
    怎么划分？
2. 算完所有功率后，汇总一个list，单线程计算阈值，并计算大于阈值的个数
    准确率是正确的除以总的
    判断条件的计算方法为
    a = np.random.randint(-5, 5, (1, 10))
    c=np.sum(a>=1)                    #条件为大于等于1
3. 将数据存入 CSV 中，保存为中间结果，应当为矩阵形式


刷牙改：
最优频带算法

找到频带宽度最小的区间
逐个迭代太慢了，于是采用二分法来收敛。
每收敛一次，同步计算区分度的差分。

初始以最小位置和最大宽度开始。宽度逐渐减小，位置均分变化？宽度最大时区分度一定是最大的，要计算的是区分度的变化。
什么情况下需要进一步优化？区分度的减小小于某个阈值，如果区分度增大那就更应该优化。

|————————|
|————|————|
|——|——|——|——|

区分度是位置和宽度的二元函数。要去找使得区分度最大的最优频带。
每次迭代中，找出区分度最大的频带
在n+1次迭代中，若出现频带横跨的情况，则可以知道，第n次迭代的最大值区分度，比n+1次最大区分度的变化量，超过某阈值。

那么说明，n频带二分法无法找到最优解。那么就四分法。四分排除法。均分四份，只除去首尾。有三种情况：去首，去尾，一起去。这样就是三线程运行。
哪种最大就要哪个。
那为啥不是一开始就四分法？
可以一开始就4分法！
下次迭代再次四分
四分找不到就八分
八分找不到就16分。
到16分就说明已经最优了。
也就是说，一旦出现当前均分数找不到的情况，下次就从该均分数开始迭代。

怎么判断区分程度？
功率是一维的，因此只需要一个阈值。
怎么判断是否收敛？区分度变化曲线的一阶差分小于某个阈值即可。

模型功率分布可以有重叠，但考虑到实际情况，连续刷有隐马尔科夫模型
功率还是要大量统计

上槽牙还要进一步测量频域特征，mfcc？

外侧面用hmm可以？
事件分为连续刷和，位1、抬起、位2
这个概率并不是通过bw算法训练出来的，而是通过模型得到的。观测值与模型值的差，以及距离上一个牙位的距离，作为转移概率？这个转移概率加一个高斯分布。
转移概率怎么算？
每个牙位，再加一个无动作的情况，共9种。两个门牙怎么办？两个门牙视为一种情况。
在当前牙区内，每种情况，转移至其他情况概率和为1。
如果切换牙区，则更换hmm，这样共4个hmm。
上侧面通过频域特征识别。

为什么不能用训练的方式？因为对用户来说，连续滑动没法找ground truth。
先说观测值与隐状态对应概率。
9种状态的功率，将功率分布划分为9个区间
对于最大值，越大越可能是模型里的最大值。
通过 1-e^x 来衡量，再向左平移一部分。
以上这个不对。
取模型点中心为1，其余位置符合正态分布？不对，因为power没有负值。待定
而且，同一个点，到模型中心的概率符合一个正态分布，到另一个模型中心符合另一个正态分布，到所有中心共有8个正态分布，和不一定为1。

转移概率：任何一个牙，自己，左右两个，加一个抬起，最大。抬起需要设置一个概率阈值，占有固定概率？以自己为中心至左右为3sigma，因为有可能滑动太快，一个窗口检测不到。
滑动时有低频分量，这样可以区分。



-
  -      -
    - - 


那么数据重新收成连续的？
前校正问题，维特比算法。那这个ground truth就自己说了算了。
按不紧的情况？

    
    """

    # path = r"C:\Users\xenon\Desktop\外侧面与上侧面\组1\上侧面.wav"
    # path = r"C:\Users\xenon\Desktop\外侧面与上侧面\组5\外侧面.wav"
    # path = r"C:\Users\xenon\Desktop\上下区分2\上\12-44-48.wav"
    # path = r"C:\Users\xenon\Desktop\原组\2.wav"

    # path = r"C:\Users\xenon\Desktop\gen\HST-DR\HST-DR-6-000.wav"
    #
    # teeth = tr.file2teeth(path)
    # print(teeth.uname)
    # predict_pos = tr.recognize_model_position(teeth)
    # print("预测位置：", predict_pos)
    # print("真实位置：", teeth.depth)
    # read = wavfile.read(path)
    # fs, signal = read
    # data_1 = signal[..., 0]
    # print("频带功率：", p.get_freq_power_by_spectrum(data_1, 497, 537))
    # v.show_am_time(data_1, fs=fs)
    # v.show_am_freq(data_1, fs=fs)

    # dc.tooth_conf_mat(r"C:\Users\xenon\Desktop\gen\HST-DL", tr.Teeth.DL)

    # dc.tooth_dist(r"C:\Users\xenon\Desktop\gen\HST-DL",
    #               r"C:\Users\xenon\Desktop\gen\HST-DR",
    #               "HST",
    #               tr.Teeth.DL,
    #               tr.Teeth.DR)

    # gen_data_single("HST", Teeth.DR, 3, src_path=r"C:\Users\xenon\Desktop\原组\2.wav")

    # dc.nature_freq(r"C:\Users\xenon\Desktop\gen\HST-UL-fake", r"C:\Users\xenon\Desktop\gen\HST-DR")

    """
    515：
    上侧面测试
    模型增加几个人
    """

    # data_1 = data_1[int(2.5 * fs):int(26 * fs)]

    # data_1 = data_1[0:int(3 * fs)]
    # data_1 = data_1[int(3 * fs):int(6 * fs)]

    # data_1 = data_1[int(0.289 * fs): int(0.318 * fs)]

    # print("1210Hz-1710Hz 功率：", p.get_freq_power_by_spectrum(data_1, 1210, 1710, fs))

    # 提取自然频率（谐波）
    # data_n = f.butter_filter(data_1, 810, 847, fs)
    # v.show_am_time(data_n, fs=fs)
    # v.show_am_freq(data_n, fs=fs)

    # data_n = 2 * f.get_mean_pooling(data_n, 2)

    # gen_data_single("LZH", Teeth.DL, 2, src_path=r"C:\Users\xenon\Desktop\原组\1.wav")

    """
    怎么调制带宽？先不管了
    先生成和模型判别代码
    """

    # if len(data_n) > len(data_1):
    #     data_n = data_n[0:len(data_1)]
    # else:
    #     data_1 = data_1[0:len(data_n)]

    # data_1 = data_1 + data_n
    # 增大频率后会缩短
    # v.show_am_freq(data_1, fs=fs)

    # uf.cut_file(path, (0, 12.7))

    """
    514功率记录：
    上
    4.2968323129350505
    4.19196568431666
    3.1644532605273743
    3.1977514194090633
    2.743357921619445
    4.7591395162948205
    
    下：
    2.507235588501252
    2.4470794551624016
    4.217734195620705
    3.0716381576139726
    1.8793726529989816
    1.354463933807899
    1.5316716140273325
    1.2613462223898653
    
    右侧：
    上：
    3.9337671717982152
    2.9134987714076135
    1.7023482477270295
    6.749852652064248 没贴紧？
    
    下：
    14.058796153731857
    2.137282254399832
    4.747363248733401
    6.832805005927035
    
    """

    # moment, power = tr.get_band_power_change(data_1, (250, 330), fs=fs)
    # moment, power = tr.get_band_power_change(data_1, (250, 330), fs=fs)
    # plt.figure(0)
    # plt.plot(moment, power)
    # plt.show()

    # src_path = r"C:\Users\xenon\Desktop\清洁3~1.wav"
    # fs, signal = rt.get_rand_len_signal(src_path, int(1.5 * 48000), int(25 * 48000))

    # v.show_csv(r"C:\Users\xenon\Desktop\slide.csv")

    # tr.gen_slide_signal()

    # uf.csv_zeroing(r"C:\Users\xenon\Desktop\slide.csv", True)

    # 批量读入的方法
    # path = r"F:\python生成的数据"
    # t_list = tr.read_as_tooth_list(path)

    # gen_data_single("LZH", Teeth.DL, 12, src_path=r"C:\Users\xenon\Desktop\原组\1.wav")
    # gen_data_single("LZH", Teeth.UL, 11, src_path=r"C:\Users\xenon\Desktop\原组\1.wav")
    # gen_data_single("HST", Teeth.DL, 13, 13, src_path=r"C:\Users\xenon\Desktop\原组\2.wav")
    # gen_data_single("HST", Teeth.UL, 11, 11, src_path=r"C:\Users\xenon\Desktop\原组\2.wav")
    # gen_data_single("MQC", Teeth.DL, 39, src_path=r"C:\Users\xenon\Desktop\原组\3.wav")
    # gen_data_single("MQC", Teeth.DR, 38, src_path=r"C:\Users\xenon\Desktop\原组\3.wav")
    # gen_data_single("SYQ", Teeth.DL, 37, src_path=r"C:\Users\xenon\Desktop\原组\4.wav")
    # gen_data_single("SYQ", Teeth.DR, 38, src_path=r"C:\Users\xenon\Desktop\原组\4.wav")

    # gen_data_single("LZH", Teeth.UL, 34, src_path=r"C:\Users\xenon\Desktop\原组\1.wav")
    # gen_data_single("LZH", Teeth.UR, 29, src_path=r"C:\Users\xenon\Desktop\原组\1.wav")
    # gen_data_single("HST", Teeth.UL, 36, src_path=r"C:\Users\xenon\Desktop\原组\2.wav")
    # gen_data_single("HST", Teeth.UR, 35, src_path=r"C:\Users\xenon\Desktop\原组\2.wav")
    # gen_data_single("MQC", Teeth.UL, 29, src_path=r"C:\Users\xenon\Desktop\原组\3.wav")
    # gen_data_single("MQC", Teeth.UR, 31, src_path=r"C:\Users\xenon\Desktop\原组\3.wav")
    # gen_data_single("SYQ", Teeth.UL, 36, src_path=r"C:\Users\xenon\Desktop\原组\4.wav")
    # gen_data_single("SYQ", Teeth.UR, 37, src_path=r"C:\Users\xenon\Desktop\原组\4.wav")

    # print(np.random.uniform(2, 5, None))

    # gen_data_single("ZR", Teeth.UL, 30)
    # gen_data_single("ZR", Teeth.UR, 30)
    # gen_data_single("SYQ", Teeth.UL, 30)
    # gen_data_single("SYQ", Teeth.UR, 30)
    # gen_data_single("SYQ", Teeth.DL, 30)
    # gen_data_single("SYQ", Teeth.DR, 30)
    # gen_data_single("HST", Teeth.UL, 30, src_path=r"C:\Users\xenon\Desktop\清洁2~1.wav")
    # gen_data_single("HST", Teeth.UR, 30, src_path=r"C:\Users\xenon\Desktop\清洁2~1.wav")
    # gen_data_single("HST", Teeth.DL, 30, src_path=r"C:\Users\xenon\Desktop\清洁2~1.wav")
    # gen_data_single("HST", Teeth.DR, 30, src_path=r"C:\Users\xenon\Desktop\清洁2~1.wav")
    # gen_data_single("MQC", Teeth.UL, 30, src_path=r"C:\Users\xenon\Desktop\清洁2~1.wav")
    # gen_data_single("MQC", Teeth.UR, 30, src_path=r"C:\Users\xenon\Desktop\清洁2~1.wav")
    # gen_data_single("MQC", Teeth.DL, 30, src_path=r"C:\Users\xenon\Desktop\清洁2~1.wav")
    # gen_data_single("MQC", Teeth.DR, 30, src_path=r"C:\Users\xenon\Desktop\清洁2~1.wav")
    # gen_data_single("ZJY", Teeth.UL, 30, src_path=r"C:\Users\xenon\Desktop\待分割\LL2.wav")
    # gen_data_single("ZJY", Teeth.UR, 30, src_path=r"C:\Users\xenon\Desktop\待分割\UR3.wav")
    # gen_data_single("ZJY", Teeth.DL, 30, src_path=r"C:\Users\xenon\Desktop\待分割\LL4.wav")
    # gen_data_single("ZJY", Teeth.DR, 30, src_path=r"C:\Users\xenon\Desktop\待分割\UR4.wav")
    # gen_data_single("LC", Teeth.UL, 30, src_path=r"C:\Users\xenon\Desktop\待分割\LL4.wav")
    # gen_data_single("LC", Teeth.UR, 30, src_path=r"C:\Users\xenon\Desktop\待分割\UR5.wav")
    # gen_data_single("LC", Teeth.DL, 30, src_path=r"C:\Users\xenon\Desktop\待分割\LL4.wav")
    # gen_data_single("LC", Teeth.DR, 30, src_path=r"C:\Users\xenon\Desktop\待分割\UR5.wav")
    # gen_data_single("JYF", Teeth.UL, 30, src_path=r"C:\Users\xenon\Desktop\待分割\LL2.wav")
    # gen_data_single("JYF", Teeth.UR, 30, src_path=r"C:\Users\xenon\Desktop\待分割\UR5.wav")
    # gen_data_single("JYF", Teeth.DL, 30, src_path=r"C:\Users\xenon\Desktop\待分割\LL2.wav")
    # gen_data_single("JYF", Teeth.DR, 30, src_path=r"C:\Users\xenon\Desktop\待分割\UR5.wav")
    # gen_data_single("WY", Teeth.UL, 30, src_path=r"C:\Users\xenon\Desktop\待分割\LL6.wav")
    # gen_data_single("WY", Teeth.UR, 30, src_path=r"C:\Users\xenon\Desktop\待分割\UR6.wav")
    # gen_data_single("WY", Teeth.DL, 30, src_path=r"C:\Users\xenon\Desktop\待分割\LL6.wav")
    # gen_data_single("WY", Teeth.DR, 30, src_path=r"C:\Users\xenon\Desktop\待分割\UR6.wav")

    # tr.nature_freq_distinguish_3()

    # data_1 = np.array(signal[..., 0], dtype=float)[int(0.16 * fs):int(0.34 * fs)]
    # data_1 = np.array(signal[..., 1], dtype=float)
    # data_2 = np.array(signal[..., 1], dtype=float)
    # data_1 = data_1[int(1.5 * fs):int(7.5 * fs)]
    # data_2 = data_2[int(1.5 * fs):int(7.5 * fs)]
    # data_1 /= 100
    # data_2 /= 100
    # data_1 = f.get_mean_pooling(data_1, 1.735)
    # data_2 = f.get_mean_pooling(data_2, 1.735)
    # power_by_diff_wnd(data_1, 285, 305)

    # dc.kalman_filter()

    # data_1 = data_1[1000:5410]
    # data_1 = signal

    # 为什么原来的就差别不大，新的就差别大？
    # print(p.get_freq_power_by_spectrum(data_1, 50, 70))

    # path = r"/Users/wangyang/Library/CloudStorage/OneDrive-个人/_真实数据/【右下】/L5.wav"

    # plt.figure()
    # plt.plot(data_2, alpha=0.8)
    # plt.plot(data_1, alpha=0.8)
    # plt.show()

    # dc.freq_domain()

    # dc.no_action()

    # dc.process_time()

    # dc.curve()

    # dc.surface()

    """
    待解释的问题：
    4410 窗口平均 1 个点 10 Hz，你怎么知道其中含有固频？
    怎么算出来的？4410 / 44100 Hz = 0.1 点/Hz
    
    多频相加问题：怎么加出频带？
    在 10Hz 区间内变频相加，最后会加在同一个尖上。
    最终会形成连续的包络曲线。
    
    左右区分：中心偏移校正
    下牙区混淆程度大一些。
    
    低采样率问题：
    波形拟合，因为只有一个峰。
    
    重要解释问题：回波分量。能量应该很高，衰减不会这么明显。
    别人也做过骨传导的，但是很显然，这种能量应该传到耳蜗位置差不多。
    
    """
    # powers = []
    # x = []
    # for i in range(0, 40000):
    #     data = [0] * (4410 + i)
    #     data = uf.add_sin(data, 1, 500, 0)
    #     data = uf.add_sin(data, 1.2, 501, 0)
    #     data = uf.add_sin(data, 1.7, 502, 0)
    #     data = uf.add_sin(data, 1.6, 503.5, 0)
    #     data = uf.add_sin(data, 1.3, 504, 0)
    #     data = uf.add_sin(data, 1.5, 505, 0)
    #     data = uf.add_sin(data, 1.9, 509, 0)
    #     data = uf.add_sin(data, 1.8, 510, 0)
    #     powers.append(p.get_freq_power_by_spectrum(data, 490, 510))
    #     x.append(4410 + i)
    #     v.show_progress(i, 40000)
    #
    # v.show_progress(1, 1)
    # plt.figure(0)
    # plt.plot(x, powers)
    # plt.show()

    # v.show_am_freq(data)

    # 小采样的条件下，不升采样都已经很高了，更何况大采样。
    # 超高采样 18.68，高采样 18.62，低采样 53.86
    # print(p.get_freq_power_by_spectrum(data, 490, 510))

    # 超高采样9.34，高采样 9.34，低采样 28.72
    # print(p.get_power(data))

    # dc.different_user_accu()

    # dc.kalman_filter()

    # dc.time_statistic_cdf()

    pass

    # 生成数据单线程版示例代码




# 计算不同窗口长度下的功率
def power_by_diff_wnd(data, low, high, start_wnd=4410, max_wnd=44100):
    powers = []
    x = []
    max_wnd -= start_wnd
    for i in range(0, max_wnd):
        current_wnd = start_wnd + i
        powers.append(p.get_freq_power_by_spectrum(data[0:current_wnd], low, high))
        # x 是窗口长度
        x.append(current_wnd)
        v.show_progress(i, max_wnd)

    v.show_progress()
    plt.figure(0)
    plt.plot(x, powers)
    plt.show()
    return
