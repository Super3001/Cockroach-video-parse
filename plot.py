# plot.py

# import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection
import numpy as np
import colorsys

def formal(x:np.ndarray, reverse=False):
    if not reverse:
        x = x/255
        x[x>1] = 1
        return x
    else:
        x = np.round(x*255)
        return np.array(x, dtype=np.int32)

'''def colorbar(start_color = np.array([255, 0, 0]), end_color = np.array([255, 255, 0]), gradient_length = 100, saturation_grad=False): # 红色，黄色，步长，饱和度渐变
    if gradient_length == 1:
        return formal(start_color)
    gradient_colors = np.zeros((gradient_length, 3))
    if saturation_grad:
        hsv = colorsys.rgb_to_hsv(*list(formal(start_color))) # 用 * 运算符将序列元素一个个作为参数传入函数
        for i in range(gradient_length):
            hsv_i = hsv[0], 1 - i/(gradient_length-1), hsv[2]
            gradient_colors[i][0], gradient_colors[i][1], gradient_colors[i][2] = colorsys.hsv_to_rgb(*hsv_i)
        return gradient_colors
    else:
        for i in range(gradient_length):
            gradient_colors[i] = (1 - i/(gradient_length-1)) * start_color + (i/(gradient_length-1)) * end_color
        return formal(gradient_colors)'''
        
def colorbar_between_two(start=np.array([255,0,0]), end=np.array([225,225,0]),length=100):
    if length == 1:
        return formal(start.reshape(1,-1))
    gradient_colors = np.zeros((length, 3))
    for i in range(length):
        gradient_colors[i] = (1 - i/(length-1)) * start + (i/(length-1)) * end
    return formal(gradient_colors)

def colorbar_saturation_grad(color=np.array([0,0,255]), descend=True, length=100):
    if length == 1:
        return formal(color)
    gradient_colors = np.zeros((length, 3))
    hsv = colorsys.rgb_to_hsv(*list(formal(color)))
    for i in range(length):
        hsv_i = hsv[0], 1 - i/(length-1), hsv[2]
        gradient_colors[i][0], gradient_colors[i][1], gradient_colors[i][2] = colorsys.hsv_to_rgb(*hsv_i)
    return gradient_colors


class Ploter:
    def __init__(self) -> None:
        pass

    def drawline(self, x, y):
        pass

    def set(self, line_color='const', point_color='const'):
        self.line_color = line_color
        self.point_color = point_color
        

    def show(self):
        pass

    def scatter(self, x, y, c=None):
        pass
        

    def savefig(self):
        pass
    