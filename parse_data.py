# parse_data.py
import time
import utils
from math import sqrt, atan, ceil, pi
import numpy as np

"""debug global property"""
from control import pstatus
# pstatus = "release"
# pstatus = "debug"

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
                    sti_sections[-1].append(self.frames.index(f))
        assert len(sti_sections) == len(self.stimulus), "Wrong Value"
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
        self.X1 = []
        self.Y1 = []
        self.frames = []
        for i in data1:
            frame, coords = i.split()
            x,y = tuple(coords.split(','))
            x=float(x)
            y=-float(y)
            frame = int(frame)
            self.X1.append(x)
            self.Y1.append(y)
            self.frames.append(frame)
            
        self.X2 = []
        self.Y2 = []
        for i in data2:
            frame, coords = i.split()
            x,y = tuple(coords.split(','))
            x=float(x)
            y=-float(y)
            self.X2.append(x)
            self.Y2.append(y)
        self.num1 = len(self.X1)
        self.num2 = len(self.X2)
        ''' 前点和后点的记录数应该都是有效读的帧数，因此...，frame可以只看其中一个'''
        assert self.num1 == self.num2, "Wrong Data"
        
        self.X_mid = []
        self.Y_mid = []
        self.K=[]
        self.D = []
        self.Theta=[]
        zerot = 0
        
        for i in range(min(len(self.X1),len(self.X2))):
            if(self.X1[i]==0 or self.X2[i]==0):
                self.frames[i] = 0
                self.frames.remove(0)
                zerot += 1
                continue
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
        
        self.frames = np.array(self.frames)

        # showinfo(message='共检测到数据'+str(len(self.X1))+' : '+str(len(self.X2)))
        self.num = len(self.frames)
        self.__available__ = ['X1','Y1','X2','Y2','X_mid','Y_mid','K','D','Theta','frames']
        # self.durings = self.sti_segment()


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
        # self.durings = self.sti_segment()

        ''' end '''

if pstatus == "debug":
    if __name__ == '__main__':
        parser = DataParser()
        parser.parse_light(open('out-light-every.txt','r'), 30)
        parser.parse_fbpoints(open('out-feature-1.txt','r'),open('out-feature-2.txt','r'),30)

