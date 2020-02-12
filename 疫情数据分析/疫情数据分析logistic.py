#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
拟合2019-nCov肺炎感染确诊人数
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def logistic_increase_function(t, K, P0, r):
    t0 = 11
    r = 0.65
    #   r = 0.55
    # t:time   t0:initial time    P0:initial_value    K:capacity  r:increase_rate
    exp_value = np.exp(r * (t - t0))
    return (K * exp_value * P0) / (K + (exp_value - 1) * P0)


'''
1.11日41例
1.18日45例
1.19日62例
1.20日291例
1.21日440例
1.22日571例
1.23日830例
1.24日1287例
1.25日1975例
1.26日2744例
1.27日4515例
'''

#  日期及感染人数
t = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
t = np.array(t)
P = [41, 45, 62, 291, 440, 571, 830, 1287, 1975, 2744, 4515]
P = np.array(P)

# 用最小二乘法估计拟合
popt, pcov = curve_fit(logistic_increase_function, t, P)
# 获取popt里面是拟合系数
print("K:capacity  P0:initial_value   r:increase_rate   t:time")
print(popt)
# 拟合后预测的P值
P_predict = logistic_increase_function(t, popt[0], popt[1], popt[2])
# 未来预测
future = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 41, 51, 61, 71, 81, 91, 101]
future = np.array(future)
future_predict = logistic_increase_function(future, popt[0], popt[1], popt[2])
# 近期情况预测
tomorrow = [28, 29, 30, 32, 33, 35, 37, 40]
tomorrow = np.array(tomorrow)
tomorrow_predict = logistic_increase_function(tomorrow, popt[0], popt[1], popt[2])

# 绘图
plot1 = plt.plot(t, P, 's', label="confimed infected people number")
plot2 = plt.plot(t, P_predict, 'r', label='predict infected people number')
plot3 = plt.plot(tomorrow, tomorrow_predict, 's', label='predict infected people number')
plt.xlabel('time')
plt.ylabel('confimed infected people number')

plt.legend(loc=0)  # 指定legend的位置右下角

print(logistic_increase_function(np.array(28), popt[0], popt[1], popt[2]))
print(logistic_increase_function(np.array(29), popt[0], popt[1], popt[2]))
plt.show()

# 未来预测绘图
# plot2 = plt.plot(t, P_predict, 'r',label='polyfit values')
plot3 = plt.plot(future, future_predict, 'r',label='polyfit values')
plt.show()


print("Program done!")
