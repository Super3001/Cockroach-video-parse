# colorbar_test.py

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

# 生成随机的数据
x = np.arange(0,10)
y = np.arange(0,10)

# 生成一个颜色数组，沿着饱和度方向渐变
colors = np.zeros((len(x), 4))
colors[:, 0] = np.linspace(0.2, 1.0, len(x))  # 设置红色通道为线性渐变
colors[:, 1] = 0.8  # 设置绿色通道为常量
colors[:, 2] = 0.2  # 设置蓝色通道为常量
colors[:, 3] = np.linspace(0.2, 1.0, len(x))  # 设置透明度为线性渐变

# 创建线段集合对象
# segments = np.array([x, y]).T.reshape(1, -1, 2)
segments = np.array([[[x[i],y[i]],[x[i+1],y[i+1]]] for i in range(len(x) - 1)])
print(segments)
lc = LineCollection(segments, colors=colors)
# lc = LineCollection(segments)

# 创建坐标轴并绘制线段集合
fig, ax = plt.subplots()
ax.add_collection(lc)

# 设置坐标轴范围
ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y), max(y))

# 显示图形
plt.show()