# -*- coding: utf-8 -*-
import json
import re

import requests
from pyecharts.charts import Map
# from win32comext.shell.demos.IActiveDesktop import opts

result = requests.get(
    'https://interface.sina.cn/news/wap/fymap2020_data.d.json?1580097300739&&callback=sinajp_1580097300873005379567841634181')
json_str = re.search("\(+([^)]*)\)+", result.text).group(1)

html = f"{json_str}"
table = json.loads(f"{html}")
# print(table)
# print(html)
'''
用pip安装pyecharts, 以及两个数据包echarts-china-provinces-pypkg
echarts-china-cities-pypkg 即可'''

data =[]
import pprint

from pyecharts import options as opts
pp = pprint.PrettyPrinter(indent=1)
for province in table['data']['list']:
    data.append((province['name'], province['value']))
    pp.pprint(province)
    # for city in province['city']:
    #     pp.pprint(city)

map_p = Map()
map_p.set_global_opts(title_opts=opts.TitleOpts(title="实时疫情图"), visualmap_opts=opts.VisualMapOpts(max_=100))
map_p.add("确诊", data, maptype="china")
# map_p.add("确诊", data[province], maptype="china")
map_p.render("ncov.html")  # 生成html文件