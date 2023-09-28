# parse_data.py
# %%
import utils
from math import sqrt, atan, ceil, pi
import numpy as np

"""debug global property"""
from control import pstatus
# pstatus = "release"
# pstatus = "debug"

""" 主要做的是：1.数据有效化 2.segment每次刺激 """
""" 数据的保存形式全部为np.ndarray """

TBD = -10

def f2sec(n,fps): # 将帧数转化为秒
    return n / fps

from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

"""
# 示例用法
fs = 100 # 采样频率
lowcut = 0.01 # 低频截止频率
highcut = 3 # 高频截止频率
order = 4 # 滤波器阶数
# 应用滤波器
filtered_data = butter_bandpass_filter(y, lowcut, highcut, fs, order=order)
"""

class DataParser:
    def __init__(self, before=0.5, after=1.5, fps=60, skip_n=1, light=True):
        self.X1 = []
        self.Y1 = []
        self.X2 = []
        self.Y2 = []
        self.X_mid = []
        self.Y_mid = []
        self.K = []
        self.D = []
        self.Theta = []
        self.__available__ = []
        self.before = before
        self.after = after
        self.frames = [] # frame numbers
        self.frames_adj = [] # adjacent to next frame

        self.has_light = light # if there is light information
        self.light_frames = [] # all the frames when the light is on
        self.stimulus = [] # all the judged stimulate frame number
        self.durings = [] # list of index(data indices) that in each stimulate section

        self.filekind = ''
        self.timestr = ''
        self.fps = fps
        self.skip_n = skip_n

    """ change all scaler of data """
    def data_change_ratio(self, ratio):
        assert ratio > 0, "Wrong Value"
        if self.filekind == 'fb':
            self.X1 = [x*ratio for x in self.X1]
            self.X2 = [x*ratio for x in self.X2]
            self.Y1 = [y*ratio for y in self.Y1]
            self.Y2 = [y*ratio for y in self.Y2]
            self.D = [d*ratio for d in self.D]

        for i in range(self.num):
            self.X_mid[i] = self.X_mid[i]*ratio
            self.Y_mid[i] = self.Y_mid[i]*ratio

        # self.K = [] # self.K 不需要修改
        # self.Theta = [] # self.Theta 不需要修改

        '''
            for each in self.__available__:
                ...
        '''

    """ return every possible frame number in each segment of stimulus, return list """
    def sti_segment(self) -> list:
        sti_sections = []
        for sti in self.stimulus:
            sti_sections.append([])
            left  = sti - ceil(self.before*self.fps)
            right = sti + ceil(self.after *self.fps)
            for f in range(left, right+1):
                if f in self.frames:
                    # sti_sections[-1].append(self.frames.index(f))
                    idx = np.where(self.frames == f)[0][0]
                    sti_sections[-1].append(idx)
        assert len(sti_sections) == len(self.stimulus), "Wrong Value"
        sti_sections = [np.array(x) for x in sti_sections] # 转为np.array，好用indice格式取数据
        return sti_sections
        

    """ 统一用frame data_format """
    def parse_light(self, file_light, fps):
        light_data = [int(x) for x in file_light.readlines()]
        stimulus = []
        for i, t in enumerate(light_data):
            if i==0:
                stimulus.append(t)
            elif f2sec(t-light_data[i-1],fps) > 1: # 间隔0.5s就算不同的刺激
                stimulus.append(t)
        self.light_frames = light_data
        self.stimulus = stimulus

    def parse_fbpoints(self,file_f,file_b, fps):
        self.filekind = 'fb'
        self.timestr = utils.timestr()

        # 声明所有可用的数据
        self.__available__ = ['X1','Y1','X2','Y2','X_mid','Y_mid','K','D','Theta','frames','frames_adj','durings']

        data1 = file_f.readlines()
        data2 = file_b.readlines()
        self.X1 = {} # {frame: value} 格式
        self.Y1 = {}
        """ X1, X2因为不是同时采样，对应的frame可能不一样 """
        nframe_1 = data1[-1].split()[0]
        nframe_1 = int(nframe_1)
        for i in data1:
            frame, coords = i.split()
            y,x = tuple(coords.split(','))
            x=float(x)
            y=-float(y)
            # frame = int(frame)
            self.X1[frame] = x
            self.Y1[frame] = y
            
        
        self.X2 = {}
        self.Y2 = {}
        nframe_2 = data2[-1].split()[0]
        nframe_2 = int(nframe_2)
        for i in data2:
            frame, coords = i.split()
            y,x = tuple(coords.split(','))
            x=float(x)
            y=-float(y)
            self.X2[frame] = x
            self.Y2[frame] = y
        self.num1 = len(self.X1)
        self.num2 = len(self.X2)
        
        self.frames = []
        self.X_mid = []
        self.Y_mid = []
        self.K=[]
        self.D = []
        self.Theta=[]
        self.zerot = 0 # 可以计算出来
        
        # 对帧
        for f in range(min(nframe_1, nframe_2)):
            i = str(f) # key也可以当做一种下标
            if i in self.X1 and i in self.X2:
                if self.X1[i]!=0 and self.X2[i]!=0:
                    """ 有效帧 """
                    self.frames.append(f)
                    xmid = (self.X1[i]+self.X2[i])/2
                    ymid = (self.Y1[i]+self.Y2[i])/2
                    self.X_mid.append(xmid)
                    self.Y_mid.append(ymid)
                    dist=sqrt((self.X2[i]-self.X1[i])*(self.X2[i]-self.X1[i]) + (self.Y2[i]-self.Y1[i])*(self.Y2[i]-self.Y1[i]))
                    if self.X2[i] - self.X1[i]==0:
                        k=0
                    else:
                        k=(self.Y2[i]-self.Y1[i])/(self.X2[i]-self.X1[i])
                    self.D.append(dist)
                    self.K.append(k)
                    self.Theta.append(atan(k)*180/pi)

        """ 这里的frames一定是统一的，而且一定是在X1, X2里有值的 """
        """ 过滤无效值 """
        self.X1 = [float(self.X1[str(f)]) for f in self.frames]
        self.Y1 = [float(self.Y1[str(f)]) for f in self.frames]
        self.X2 = [float(self.X2[str(f)]) for f in self.frames]
        self.Y2 = [float(self.Y2[str(f)]) for f in self.frames]
        
        """ 全部转为np.array """
        self.frames = np.array(self.frames)
        self.num = len(self.frames)
        assert self.num > 0, "Data error"
        self.frames_adj= np.zeros((self.num-1,))
        for i in range(self.num-1):
            if self.frames[i+1] - self.frames[i] <= self.skip_n:
                self.frames_adj[i] = 1
        
        self.X1 = np.array(self.X1); self.Y1 = np.array(self.Y1); self.X2 = np.array(self.X2); self.Y2 = np.array(self.Y2)
        self.X_mid = np.array(self.X_mid); self.Y_mid = np.array(self.Y_mid)
        self.K = np.array(self.K); self.D = np.array(self.D); self.Theta = np.array(self.Theta)
        self.smooth_thetas(90)
        
        if self.has_light:
            self.durings = self.sti_segment()
        
    def parse_feature_result(self, file_f,file_b, fps):
        self.filekind = ['f','b']
        self.parse_time = utils.timestr()
        self.__available__ = ['X1','Y1','X2','Y2','X_mid','Y_mid','K','D','Theta','frames','assist_angle_f','assist_angle_b','frames_adj','frame_interval','durings'] # 所有可供使用的数据

        data_f = file_f.readlines()
        data_b = file_b.readlines()
        self.X1 = {}
        self.Y1 = {}
        self.assist_angle_f = {}

        frame = 0
        # 逐行解析
        for line in data_f:
            items = line.split()
            frame = items[0]
            if items[1] == 'angle': # if len(items) == 3:
                self.assist_angle_f[frame] = items[2]
            elif 'black' in items[1] or 'relocate' in items[1]:
                continue
            else:
                y, x = items[1].split(',')
                x = float(x)
                y = -float(y)
                self.X1[frame] = x
                self.Y1[frame] = y

        self.X2 = {}
        self.Y2 = {}
        self.assist_angle_b = {}

        # 逐行解析后点
        for line in data_b:
            items = line.split()
            frame = items[0]
            if items[1] == 'angle': # if len(items) == 3:
                self.assist_angle_b[frame] = items[2]
            elif 'black' in items[1] or 'relocate' in items[1]:
                continue
            else:
                y, x = items[1].split(',')
                x = float(x)
                y = -float(y)
                self.X2[frame] = x
                self.Y2[frame] = y

        # 对帧
        """nframe表示最大的帧, num表示帧数量"""
        nframe_1 = int(data_f[-1].split()[0])
        nframe_2 = int(data_b[-1].split()[0])

        self.num1 = len(self.X1)
        self.num2 = len(self.X2)

        '''以下都使用np.array格式'''
        self.frames = []
        self.zerot = TBD # 无效帧的数量

        # 重要: range记得+1
        for f in range(min(nframe_1, nframe_2) + 1):
            i = str(f) # key should be str
            if i in self.X1 and i in self.X2:
                self.frames.append(f)

        self.frames = np.array(self.frames) # 表示有效帧
        
        # 计算帧特征
        self.nframe = np.max(self.frames)
        self.num = self.frames.size
        self.zerot = min(self.num1, self.num2) - self.num
        print('num:', self.num, 'nframe:', self.nframe, 'zerot:', self.zerot)
        
        # 提取中心点
        self.X_mid = []
        self.Y_mid = []

        for f in self.frames:
            i = str(f)
            xmid = (self.X1[i]+self.X2[i])/2
            ymid = (self.Y1[i]+self.Y2[i])/2
            self.X_mid.append(xmid)
            self.Y_mid.append(ymid)
        self.X_mid = np.array(self.X_mid); self.Y_mid = np.array(self.Y_mid)
        
        # 提取（计算）角度
        self.D = []
        self.K = []
        self.Theta = []

        for f in self.frames:
            i = str(f)
            point_f = np.array([self.X1[i], self.Y1[i]])
            point_b = np.array([self.X2[i], self.Y2[i]])
            dist = np.linalg.norm(point_b - point_f)
            dx, dy = point_b - point_f # numpy.array解包
            if abs(dx) < 1e-6: # 竖直
                if abs(dy) < 1e-6:
                    raise Exception("前后点重合")
                k = np.inf
                theta = 90
            else:
                k = - dy / dx # 因为之前 y = -y
                theta = atan(k)*180/pi
            
            self.D.append(dist)
            self.K.append(k)
            self.Theta.append(theta) 
        self.K = np.array(self.K); self.D = np.array(self.D); self.Theta = np.array(self.Theta)

        # 注：theta要连续，而atan生成的不连续，所以要处理
        self.smooth_thetas(90) # 90度以上认为是跳变

        # 过滤出有效值（并转为np.array, dtype=np.float）
        """这里的frames一定是统一的，而且一定是在X1, X2里有值的"""
        self.X1 = np.array([float(self.X1[str(f)]) for f in self.frames])
        self.Y1 = np.array([float(self.Y1[str(f)]) for f in self.frames])
        self.X2 = np.array([float(self.X2[str(f)]) for f in self.frames])
        self.Y2 = np.array([float(self.Y2[str(f)]) for f in self.frames])
        self.assist_angle_f = np.array([float(self.assist_angle_f[str(f)]) for f in self.frames])
        self.assist_angle_b = np.array([float(self.assist_angle_b[str(f)]) for f in self.frames])

        # 计算帧近邻
        assert self.num > 0, "Data error"
        # self.frames_adj = np.zeros((self.num-1,), dtype=np.bool)
        # 注意：np.where返回的是tuple
        self.frames_adj = np.where(self.frames[1:] - self.frames[:-1] <= self.skip_n)[0]

        # 计算帧间隔
        # self.frame_interval = np.zeros((self.num-1,), dtype=np.int)
        self.frame_interval = self.frames[1:] - self.frames[:-1]

        # 分割刺激
        self.smooth_thetas(90) # 90度以上认为是跳变
        
        if self.has_light:
            self.durings = self.sti_segment()
        
        ''' end '''

    def smooth_thetas(self,threshold: np.array):
        smoothed_thetas = self.Theta.copy()
        for i in range(1, len(smoothed_thetas)):
            diff = smoothed_thetas[i] - smoothed_thetas[i-1]
            if diff > threshold: # 由负跳变为正
                smoothed_thetas[i:] -= 180
            elif diff < -threshold: # 由正跳变为负
                smoothed_thetas[i:] += 180
        thetas = smoothed_thetas - smoothed_thetas[0] # 以第一个角度为0
        self.Theta = thetas

    def parse_center_angle(self, file_center,file_angle,fps):
        self.filekind = 'ca'
        self.timestr = utils.timestr()
        self.__available__ = ['X_mid','Y_mid','Theta','frames']

        data1 = file_center.readlines()
        data2 = file_angle.readlines()
        self.X1 = []
        self.Y1 = []
        self.X2 = []
        self.Y2 = []
        self.K = []
        self.D = []
        """ 重置：防止污染数据 """
        
        """ center和angle是同时记录的 """
        self.X_mid = []
        self.Y_mid = []
        for i in data1:
            frame, x, y = i.split()
            x = x[:-1] # (x, y)格式，去掉','
            x=float(x)
            y=-float(y)
            frame = int(frame)
            self.X_mid.append(x)
            self.Y_mid.append(y)
            self.frames.append(frame)
        self.num1 = len(self.X_mid)
            
        self.Theta = []
        for i in data2:
            frame, theta = i.split()
            theta = float(theta)
            self.Theta.append(theta)
        self.num2 = len(self.Theta)
        # assert self.num1 == self.num2, "Wrong Data"

        self.nframe = self.num1

        # 对帧
        """ 有indice和mask两种写法，考虑到被过滤掉的数据少于保留的数据，这里采用mask的写法"""
        mask = np.ones((self.num1,))
        for i in range(self.num1):
            if self.X_mid[i] == 0 or self.Y_mid[i] == 0:
                mask[i] = 0
                # 标记为非法数据，mask记录下标
        self.X_mid = np.array(self.X_mid)[mask>0]
        self.Y_mid = np.array(self.Y_mid)[mask>0]
        self.Theta = np.array(self.Theta)[mask>0]
        self.frames = np.array(self.frames)[mask>0]
        self.num = len(self.frames)
        assert self.num > 0, "Data error"
        self.frames_adj = np.zeros((self.num-1,))
        for i in range(self.num-1):
            if self.frames[i+1] - self.frames[i] <= self.skip_n:
                self.frames_adj[i] = 1

        self.smooth_thetas(90) # 90度以上认为是跳变
        
        if self.has_light:
            self.durings = self.sti_segment()

        ''' end '''

# %%
if pstatus == "debug":
    if __name__ == '__main__':
        parser = DataParser()
        parser.parse_light(open('out-light-every.txt','r'), 30)
        parser.parse_feature_result(open('out-feature-1.txt','r'),open('out-feature-2.txt','r'),30)
        
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(15, 14))
        plt.subplot(2, 1, 1)
        plt.plot(parser.frames, parser.Theta)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title('Original Data')

        # 示例用法
        fs = 60 # 采样频率
        lowcut = 0.01 # 低频截止频率
        highcut = 3 # 高频截止频率
        order = 4 # 滤波器阶数
        # 应用滤波器
        y = np.zeros(parser.nframe)
        y[parser.frames-1] = parser.Theta
        filtered_data = butter_bandpass_filter(y, lowcut, highcut, fs, order=order)
        
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(1,parser.nframe+1), filtered_data)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title('filtered')
        
        plt.show()
        print('finish')

