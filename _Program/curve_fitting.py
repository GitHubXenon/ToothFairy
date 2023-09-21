# 曲线拟合实验
import numpy as np
import utility_functions as uf
import view as v
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

fitting_x = np.array(np.arange(1, 8.1), dtype=int)
# 增量 500 700 1000 1500 2500 4500 7500
# fitting_y = np.array([2000, 2500, 3200, 4200, 5700, 8200, 12700, 20200])
# fitting_y = np.array([47000, 42000, 37000, 33000, 30000, 26598, 22000, 18000])
fitting_y = np.array([14000, 13000, 11000, 10000, 9500, 9038.91, 8000, 7500])
# fitting_y += 3000

# 从 CSV 中读取数据
# fitting_x, fitting_y = uf.csv2xy(r"C:\Users\xenon\Desktop\假设真实值.csv")

# func 指的是 CURVE 或 SHOW 中的函数
# 下牙区指定参数
# func = uf.depth_distinguish_lower
# param_bounds = ((0.05, 0.05, 2), (0.5, 0.5, 10))

# 上牙区指定参数
func = uf.depth_distinguish_upper
param_bounds = ((0.1, 0.05, 0.15, 0.1), (0.5, 0.5, 0.5, 1))

# 显示用指定参数
# func = uf.spherical_attenuation_show


"""
现在怎么拟合？
先找几个用户数据，然后改成数据中的样子
plnx + qx + d 中，q 影响最大，q 增大，差距增大

上牙区主要问题：
1位 太高，且有弯曲

增大 h 可以降低 1位
增大 q 降低 1位弯折 但后续弯折也会加深
增大 d，整体降低

"""

# 在这里定义待拟合的参数
p_opt, p_cov = curve_fit(func, fitting_x, fitting_y, bounds=param_bounds)
p_opt = np.around(p_opt, 2)
p_opt = [0.17, 0.18, 0.39, 0.27]
print("拟合的参数为：")
print(uf.list2str(p_opt))

# 查看拟合曲线
# 准备采样参数
proc_range = [0.8, 8.5]
step = (proc_range[1] - proc_range[0]) / 500

# 采样
x, y = v.get_an_samp(func, proc_range, step, p_opt, show_prog=True)

# 此处若报错，请检查是否修改 func

# 画图
plt.figure(0)
plt.plot(x, y, color='blue')
plt.scatter(fitting_x, fitting_y, marker='x', color='orange', label='fitting data')
plt.grid(True)
plt.legend()
plt.xticks(fitting_x, fitting_x)
plt.show()

# 代入通过拟合曲线，计算出的离散值（控制台显示），或者说依据拟合曲线重新采样的点
# 用于指定参数后微调，或者获取标准值
proc_range = [1, len(fitting_x)]
step = 1
x, y = v.get_an_samp(func, proc_range, step, p_opt, show_prog=True)
y = np.around(y, 2)
print("x =", uf.list2str(x))
print("y =", uf.list2str(y))
