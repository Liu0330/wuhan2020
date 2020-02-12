from scipy.optimize import curve_fit
import urllib
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import scipy as sp
from scipy.stats import norm


def date_encode(date):
    # '01.24' -> 1 * 100 + 24 = 124
    d = date.split('.')
    month, day = int(d[0]), int(d[1])
    return 100 * month + day


def date_decode(date):
    # 124 -> '01.24'
    return '{}.{}'.format(str(date // 100), str(date % 100))


def sequence_analyse(data):
    date_list, confirm_list, dead_list, heal_list, suspect_list = [], [], [], [], []
    data.sort(key=lambda x: date_encode(x['date']))
    for day in data:
        date_list.append(date_encode(day['date']))
        confirm_list.append(int(day['confirm']))
        dead_list.append(int(day['dead']))
        heal_list.append(int(day['heal']))
        suspect_list.append(int(day['suspect']))
    return pd.DataFrame({
        'date': date_list,
        'confirm': confirm_list,
        'dead': dead_list,
        'heal': heal_list,
        'suspect': suspect_list
    })


def get_date_list(month):
    month_day = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    ans = []
    for i in range(1, month_day[month - 1] + 1):
        if month == 1 and i < 13:
            continue
        ans.append(100 * month + i)
    return np.array(ans)


url = 'https://view.inews.qq.com/g2/getOnsInfo?name=wuwei_ww_cn_day_counts'
response = urllib.request.urlopen(url)
json_data = response.read().decode('utf-8').replace('\n', '')
data = json.loads(json_data)
data = json.loads(data['data'])

df = sequence_analyse(data)
x, y = df['date'].values[:-1], df['confirm'].values[:-1]
x_idx = list(np.arange(len(x)))


def func(x, a, b, c):
    return a * np.exp(b * x) + c


def f_3(x, A, B, C, D):
    return A * x * x * x + B * x * x + C * x + D


def f_4(x, A, B, C, D, E):
    return A * x * x * x * x + B * x * x * x + C * x * x + D * x + E


plt.figure(figsize=(15, 8))
plt.scatter(x, y, color='purple', marker='x', label="History data")
plt.plot(x, y, color='gray', label="History curve")
popt, pcov = curve_fit(func, x_idx, y)

test_x = x_idx + [i + 2 for i in x_idx[-2:]]
label_x = np.array(test_x) + 113
test_y = [func(i, popt[0], popt[1], popt[2]) for i in test_x]
plt.plot(label_x, test_y, 'g--', label="Fitting curve")
plt.title("{:.4}·e^{:.4}+({:.4})".format(popt[0], popt[1], popt[2]), loc="center", pad=-40)
plt.scatter(label_x[-2:], test_y[-2:], marker='x', color="red", linewidth=7, label="Predicted data")
plt.xticks(label_x, [date_decode(i) for i in label_x])
plt.ylim([-500, test_y[-1] + 2000])
plt.legend()

for i in range(len(x)):
    plt.text(x[i], test_y[i] + 200, y[i], ha='center', va='bottom', fontsize=12, color='red')
for a, b in zip(label_x, test_y):
    plt.text(a, b + 800, int(b), ha='center', va='bottom', fontsize=12)