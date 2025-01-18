# plotter.py
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
