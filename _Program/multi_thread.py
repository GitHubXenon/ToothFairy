"""
多线程并行模块：
往往都是在循环中出现的性能损失。
1. 数据均分算法
2. 将多线程任务封装成一个函数，输入总数据列表和处理函数，等待返回即可。
    要求列表中前后数据不能有关联。即可以均分处理。
"""
import math
import types
from multiprocessing import cpu_count
import threading
import time
import utility_functions as uf
import inspect
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np

"""
    【关于函数对象传参问题】
    可变长度形参： *args 和 **kwargs
    其中 *args 是元组形式传入，**kwargs 是键值对形式传入。
    获取参数列表的方法： https://www.moonapi.com/news/2566.html

        获取函数参数列表的方法
            params = inspect.signature(func).parameters
        利用获取的参数列表传参的方法
        方法一：匿名参数列表法
            param_vals = [1, 2, 3]
            print(func(*params_vals))
        方法二：命名参数字典法
            param_vals={}
            for p in params:
                param_vals[p] = 1
            print(func(**param_vals))
            
        params 中每一项的数据结构是：
            "变量名字符串" : <Parameter 类>
            Parameter 类输出时，其内容是该变量的变量名或 "变量名=变量值" 的形式。
            
    【多核运行用法示例】
        process = mp.Process(target = myFunc, args=(i,))
        参数分别为待调用函数、参数列表。
        process.start()
        process.join()
        与自写的多线程不同，该方法省略了参数类型，并且直接就能返回值。
        
        # 引入进程池
        from multiprocessing import Pool
        # 给定子进程个数
        pool = Pool(processes=4)
        # 存储结果
        results = []
        start = time.time()
        
        for n in lists:
            # 顺次启动进程，add 是功能函数。参数列表以元组形式按位置排列输入。
            # apply_async 是 apply 的并行变体，使用 apply 会造成队列阻塞，没法实现并行。
            # 返回值 result 是个对象，含有空值。只有当线程执行结束后，其值才为结果值，因此需要在 join 之后调用 get 方法获取。
            result = pool.apply_async(add, (n,) 
            # 把返回的结果添加到 results 中
            results.append(result)
            
        pool.close() # 关闭进程池，不能再添加进程
        pool.join() # 主进程等待所有进程执行完毕。所有进程执行完毕后，result 才统一返回结果。
        print(f'multi-process time cost: {time.time() - start}s')
        
        #获取结果
        for res in results:
            print(res.get())

        原文链接：https://blog.csdn.net/weixin_41099877/article/details/107956679
"""

ARG_INVALID = 0
ARG_DICT = 1
ARG_LIST = 2


def mt_list2list(func, args, data_pos, thread_num=0):
    """
    通过多线程方式，将一个 list 处理为另一个 list
    |---|---|---| -> |==|==|==|
    输入 list 可以会被平均拆分给多个线程，计算结果汇总
    :param func: 处理函数
    :param args: 传入数据必须为 (1) list 类型 (2) 键值对 dict 类型
    :param data_pos: 在函数输入参数 args 中，需要分割的数据是第几个参数（args 中包含了数据）
    :param thread_num: 线程数
    :return: 数据的输入输出均为 list
    """
    # 起始时间，注意 time 的单位是秒，不是毫秒！
    t_start = time.time()

    # 基本校验：校验第一个是否为函数
    # 这里获取 args 结果没有用，因为不论 dict 还是 list 都可以用 [] 获取值
    is_valid, arg_type = arg_check(func, args, data_pos)
    if not is_valid:
        return 0
    # 这里的 func 是形参
    # 获取参数列表
    data_list = args[data_pos]
    # 默认线程数为 CPU 核心数
    if thread_num <= 0:
        thread_num = cpu_count()
    # 规定线程数不能超过数据长度
    if thread_num > len(data_list):
        thread_num = len(data_list)

    pool = Pool(thread_num)
    print("已启用 [", thread_num, "] 个线程。")

    # 数据矩阵
    data_mat = multi_division(data_list, thread_num)

    # 进度矩阵（因为可变参数必须为列表）
    # 进度矩阵由所有线程共享，也就是说，每个线程都知道其他线程的进度。
    # 该列表的最后一位表示总数
    proc_list = [0] * thread_num
    proc_list.append(len(data_list))

    # 进程返回值列表
    result_list = []

    for i in range(thread_num):
        # 重新构造参数对象
        cur_arg = args
        cur_arg[data_pos] = data_mat[i]
        cur_arg.append(i)
        cur_arg.append(proc_list)
        # 添加进程并开始运行
        result_list.append(pool.apply_async(func, cur_arg))
    # 关闭进程池，不再添加进程。
    print("进程已添加完成。")
    pool.close()
    pool.join()
    result = []
    # 报错：pickle 不能获得内部类，例如 lambda 表达式函数等。
    for r in result_list:
        # append 是整体直接追加（list 作为一个元素），extend 是值追加（将被追加的列表拆分）
        # result.append(r.get())
        result.extend(r.get())

    print("多线程结束，总用时 [", round(time.time() - t_start, 3), "] 秒")
    return result


# 参数校验
# 返回值有两个，一个是是否合法的布尔值，另一个是参数类型。
def arg_check(func, args, data_position):
    if not isinstance(func, types.FunctionType):
        print("错误！参数请输入函数！")
        return False, ARG_INVALID
    params = inspect.signature(func).parameters
    if uf.is_dict(args):
        # 两个字典的双向验证
        for a in args:
            if not (a in params):
                print("错误！输入参数中存在函数不存在的参数！")
                return False, ARG_INVALID
        for p in params:
            if not (p in args):
                print("错误！输入参数中缺少函数规定的参数！")
                return False, ARG_INVALID
        if not (data_position in args):
            print("错误！数据键输入不正确，未在参数列表中找到！请检查是否输入了数字索引或其他内容。")
            return False, ARG_INVALID
        return True, ARG_DICT
    if uf.is_ary(args):
        # 注意：参数列表可以长度不一致，因为有缺省参数。（默认值）
        # if len(args) != len(params):
        #     print("错误！输入参数列表与函数参数列表长度不一致！")
        #     return False, ARG_INVALID
        if not uf.is_num(data_position):
            print("错误！索引位置请输入数字！")
            return False, ARG_INVALID
        return True, ARG_LIST
    # 如果参数不属于以上两种情况，同样判定为无效参数。
    print("错误！参数非 list 或 dict 类型！")
    return False, ARG_INVALID


# 线程任务均分算法
def multi_division(data_list, num):
    if not uf.is_ary(data_list):
        print("错误！输入的数据不是列表！")
        return 0
    else:
        # 单个列表长度，向上取整。
        each_len = math.ceil(len(data_list) / num)
        data_mat = []
        start = 0
        while True:
            if len(data_list) - start - 1 <= each_len:
                data_mat.append(data_list[start:])
                break
            else:
                data_mat.append(data_list[start:start + each_len])
                start += each_len
    return data_mat


class ProcessThread(threading.Thread):
    def __init__(self, func, args, arg_type):
        threading.Thread.__init__(self)
        if not isinstance(func, types.FunctionType):
            print("错误！参数请输入函数！")
        else:
            self.func = func
            self.args = args
            self.arg_type = arg_type
            self.result = []

    # run 函数本身不能返回值
    def run(self):
        if self.arg_type == ARG_DICT:
            self.result = self.func(**self.args)
        elif self.arg_type == ARG_LIST:
            self.result = self.func(*self.args)
        else:
            print("错误！线程输入参数类型不正确！")
        return

    # 该方法有两个用途：一个是用于返回值，另一个是线程等待。外部调用该方法可直接获得返回值。
    # 调用线程执行该方法时，会在调用位置阻塞，直到该方法执行完。join 翻译成 "汇合"。
    def join(self, **kwargs):
        super().join()
        return self.result
