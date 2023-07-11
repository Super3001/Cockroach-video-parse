# parse_data.py
import time
import utils
from math import sqrt, atan, ceil, pi
import numpy as np

"""debug global property"""
from control import pstatus
# pstatus = "release"
# pstatus = "debug"

""" 主要做的是：1.数据有效化 2.segment每次刺激 """
""" 数据的保存形式全部为np.ndarray """

def f2sec(n,fps): # 将帧数转化为秒
    return n / fps

class DataParser:
    def __init__(self, before=0.5, after=4.5, fps=60) -> None:
        self.X1 = []
        self.Y1 = []
        self.X2 = []
        self.Y2 = []
        self.X_mid = []
        self.Y_mid = []
        self.K = []
        self.D = []
        self.Theta = []
        self.__all__ = ['X1','Y1','X2','Y2','X_mid','Y_mid','K','D','Theta', 'frames']
        self.__available__ = []
        self.before = before
        self.after = after
        self.frames = [] # frame numbers

        self.light_frames = [] # all the frames when the light is on
        self.stimulus = [] # all the judged stimulate frame number
        self.durings = [] # list of index(data indices) that in each stimulate section

        self.filekind = ''
        self.timestr = ''
        self.fps = fps

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
                    idx = np.where(self.frames == f)[0]
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
            elif f2sec(t-light_data[i-1],fps) > 0.5:
                stimulus.append(t)
        self.light_frames = light_data
        self.stimulus = stimulus

    def parse_fbpoints(self,file_f,file_b, fps):
        self.filekind = 'fb'
        self.timestr = utils.timestr()
        data1 = file_f.readlines()
        data2 = file_b.readlines()
        self.X1 = {} # {frame: value} 格式
        self.Y1 = {}
        """ X1, X2因为不是用时采样，对应的frame可能不一样 """
        nframe_1 = data1[-1].split()[0]
        nframe_1 = int(nframe_1)
        for i in data1:
            frame, coords = i.split()
            x,y = tuple(coords.split(','))
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
            x,y = tuple(coords.split(','))
            x=float(x)
            y=-float(y)
            self.X2[frame] = x
            self.Y2[frame] = y
        self.num1 = len(self.X1)
        self.num2 = len(self.X2)
        ''' 前点和后点的记录数应该都是有效读的帧数，因此...，frame可以只看其中一个'''
        assert self.num1 == self.num2, "Wrong Data"
        
        self.frames = []
        self.X_mid = []
        self.Y_mid = []
        self.K=[]
        self.D = []
        self.Theta=[]
        zerot = 0 # 可以计算出来
        
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
        self.X1 = np.array(self.X1); self.Y1 = np.array(self.Y1); self.X2 = np.array(self.X2); self.Y2 = np.array(self.Y2)
        self.X_mid = np.array(self.X_mid); self.Y_mid = np.array(self.Y_mid)
        self.K = np.array(self.K); self.D = np.array(self.D); self.Theta = np.array(self.Theta)
        self.num = len(self.frames)
        self.__available__ = ['X1','Y1','X2','Y2','X_mid','Y_mid','K','D','Theta','frames']
        self.durings = self.sti_segment()


    def parse_center_angle(self, file_center,file_angle,fps):
        self.filekind = 'ca'
        self.timestr = utils.timestr()
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
            frame, coords = i.split()
            x,y = tuple(coords.split(','))
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
        assert self.num1 == self.num2, "Wrong Data"

        # 数据过滤
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

        # showinfo(message='共检测到数据'+str(len(self.X_mid)))
        self.num = len(self.frames)
        self.__available__ = ['X_mid','Y_mid','Theta','frames']
        self.durings = self.sti_segment()

        ''' end '''

if pstatus == "debug":
    if __name__ == '__main__':
        parser = DataParser()
        parser.parse_light(open('out-light-every.txt','r'), 30)
        parser.parse_fbpoints(open('out-meanshift-1.txt','r'),open('out-meanshift-2.txt','r'),30)
        print('finish')

