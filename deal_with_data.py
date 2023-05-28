import numpy as np
from math import *
from tkinter.messagebox import *
from scipy import interpolate
import matplotlib.pyplot as plt
import datetime
import utils
import matplotlib
matplotlib.use('TkAgg')

"""debug global property"""
from control import pstatus
# pstatus = "release"
# pstatus = "debug"

# global exception exit
utils.set_exit()

# global strings:
black_sign = '-'*5 + ' black ' + '-'*5

def f2sec(n,fps): # 将帧数转化为秒
        return n / fps
    
def interpolate_b_spline(x, y, x_new, der=0):
    """ B 样条曲线插值 或者导数. 默认der = 0"""
    tck = interpolate.splrep(x, y)
    y_bspline = interpolate.splev(x_new, tck, der=der)
    # print(y_bspline)
    return y_bspline

def Bessel_curve(control_points,tList):
    n = len(control_points)-1
    inter_points=[]
    for t in tList:
        Bp = np.zeros(2,np.float64)
        for i in range(len(control_points)):
            Bp = Bp + comb(n,i) * pow(1-t,n-i) * pow(t,i) * np.array(control_points[i])
        inter_points.append(list(Bp))
    return inter_points

def show_curve(points, tList, ):
    interPointsList = Bessel_curve(points,tList)
    x = np.array(interPointsList)[:,0]
    y = np.array(interPointsList)[:,1]
    plt.figure()
    plt.plot(x,y,color='b')
    plt.scatter(np.array(points)[:,0],np.array(points)[:,1],color='r')
    plt.title('interpolate curve')
    plt.savefig('interpolate curve demo.png')
    plt.show()
    
def show_der(points, tList):
    interPointsList = Bessel_curve(points,tList)
    x = np.array(interPointsList)[:,0]
    y = np.array(interPointsList)[:,1]
    y_der = np.zeros(len(x),np.float64)
    for i in range(1,len(x)-1):
        y_der[i] = (y[i+1]-2*y[i]+y[i-1])/(x[i+1]-x[i-1])
    plt.figure()
    plt.plot(x[1:-1],y_der[1:-1],color='b')
    plt.title('interpolate curve derivative')
    plt.savefig('interpolate curve derivative demo.png')
    plt.show()

class Dealer:
    
    def __init__(self,cap=None,filename=None,root=None,progressBar=None,markstr=None) -> None:
        print(cap)
        if cap!=None:
            self.cap = cap
            self.fps = int(cap.get(5))
            self.root = root
            self.progressBar = progressBar
        else:
            self.cap = None
            self.fps = 30
            self.root = root
            self.progressBar = progressBar
        self.X1 = []
        self.Y1 = []
        self.X2 = []
        self.Y2 = []
        self.K = []
        self.D = [] # format: [val]
        """changed:"""
        self.X_mid = [] # format: [[frame, val]]
        self.Y_mid = []
        self.Theta = []
        self.before = 0.5
        self.after = 4.5
        self.light_frames = [] # all the frames when the light is on
        self.lighttime = [] # all the judged stimulate frame number
        self.stimus = [] # list of frames that in each stimulate section
        self.frames = [] # frame numbers
        now = datetime.datetime.now()
        self.timestr = now.strftime("%Y-%m-%d-%H-%M-%S")
        # self.timestr = markstr # keep the interface with accordance
        self.filename = filename
        self.out_ratio = 0
        self.str_scale = 'px'

    def To_centimeter(self, ratio):
        self.out_ratio = ratio
        self.str_scale = 'cm'
        
    def To_origin(self):
        self.out_ratio = 0
        self.str_scale = 'px'
        
    """change all scaler of data"""
    def data_change_ratio(self, ratio):
        assert ratio > 0, "Wrong Value"
        self.X1 = [x*ratio for x in self.X1]
        self.X2 = [x*ratio for x in self.X2]
        self.Y1 = [y*ratio for y in self.Y1]
        self.Y2 = [y*ratio for y in self.Y2]
        # self.K = []
        self.D = [d*ratio for d in self.D]
        for i in range(self.num):
            self.X_mid[i][1] = self.X_mid[i][1]*ratio
            self.Y_mid[i][1] = self.Y_mid[i][1]*ratio
        # self.Theta = []

    """changed: self.stimus"""
    def parse_fbpoints(self,file_f,file_b, fps):
        data1 = file_f.readlines()
        data2 = file_b.readlines()
        self.X1 = []
        self.Y1 = []
        cnt = 1
        for i in data1:
            x,y = tuple(i.split(', '))
            x=float(x)
            y=-float(y)
            self.X1.append(x)
            self.Y1.append(y)
            cnt+=1
            
        self.X2 = []
        self.Y2 = []
        cnt = 1
        for i in data2:
            x,y = tuple(i.split(', '))
            x=float(x)
            y=-float(y)
            self.X2.append(x)
            self.Y2.append(y)
            cnt+=1
        
        # showinfo(message='共检测到数据'+str(len(self.X1))+' : '+str(len(self.X2)))
        self.K=[]
        self.Theta=[]
        zerot = 0
        
        for i in range(min(len(self.X1),len(self.X2))):
            if(self.X1[i]==0 or self.X2[i]==0):
                zerot += 1
                continue
            xmid = (self.X1[i]+self.X2[i])/2
            ymid = (self.Y1[i]+self.Y2[i])/2
            self.X_mid.append([i,xmid])
            self.Y_mid.append([i,ymid])
            dist=sqrt((self.X2[i]-self.X1[i])*(self.X2[i]-self.X1[i]) + (self.Y2[i]-self.Y1[i])*(self.Y2[i]-self.Y1[i]))
            if self.X2[i] - self.X1[i]==0:
                k=0
            else:
                k=(self.Y2[i]-self.Y1[i])/(self.X2[i]-self.X1[i])
            self.D.append(dist)
            self.K.append(k)
            self.Theta.append(atan(k)*180/pi)
        
        self.frames = [i[0] for i in self.X_mid]
        self.num = len(self.frames)
        self.stimus = self.sti_segment()
        pass
                    
    def parse_center_angle(self, file_center,file_angle,fps):
        data1 = file_center.readlines()
        data2 = file_angle.readlines()
        self.X_mid = []
        self.Y_mid = []
        cnt = 0
        for i in data1:
            x,y = tuple(i.split(', '))
            x = float(x)
            y = -float(y)
            self.X_mid.append([cnt,x])
            self.Y_mid.append([cnt,y])
            cnt+=1
            
        self.Theta = []
        cnt = 0
        for i in data2:
            theta = float(i)
            self.Theta.append(theta)
            cnt += 1
        # showinfo(message='共检测到数据'+str(len(self.X_mid)))
        self.frames = [i[0] for i in self.X_mid]
        self.num = len(self.frames)
        self.stimus = self.sti_segment()

    """bool data_format"""            
    def deal_time(self,file_light, fps):
        light_data = file_light.readlines()
        self.lighttime = []
        flag = 0
        last_light = 0
        for i, line in enumerate(light_data):
            if line[0] == '1':
                if flag == 0:
                    flag = 1
                elif flag == 1:
                    self.lighttime.append(last_light)
                    flag = 2
                last_light = i
            else:
                if f2sec(i-last_light,fps)>0.5:
                    flag = 0
    
    """frame data_format"""
    def deal_time(self, file_light, fps):
        light_data = [int(x) for x in file_light.readlines()]
        lighttime = []
        for i, t in enumerate(light_data):
            if i==0:
                lighttime.append(t)
            elif f2sec(t-light_data[i-1],fps) > 0.5:
                lighttime.append(t)
        self.light_frames = light_data
        self.lighttime = lighttime

    def minDis(self,f):
        min = 1e6
        for i in self.lighttime:
            if abs(f - i)<min:
                min = abs(f-i)
        return min
    
    """pf: self.frames[pf], pt: self.lighttime[pt]"""
    def in_section(self, pf, pt):
        frame = self.frames[pf]
        stimulus = self.lighttime[pt]
        left  = stimulus - ceil(self.before*self.fps)
        right = stimulus + ceil(self.after *self.fps)
        if frame >= left and frame <= right:
            return True
        return False
        
    """@depricated: judge if a frame number in section of stimulate"""
    def in_range(self,f):
        for i in self.lighttime:
            if f-i >= -0.5*self.fps and f-i <= 4.5*self.fps:
                return True
        return False
    
    """origin line segment in one plot"""
    def segment(self,down,up):
        for i in self.lighttime:
            x,y = [i-0.5*self.fps,i-0.5*self.fps],[down,up]
            plt.plot(x,y,color="navy")
            x,y = [i+4.5*self.fps,i+4.5*self.fps],[down,up]
            plt.plot(x,y,color="red")
    
    """@depricated: segment and plot in one function"""        
    def segment_plt(self,data:list,xlabel,ylabel,name,colors=['b','y']):
        plt.figure()
        num_stimulate = len(self.lighttime)
        for i, stimulus in enumerate(self.lighttime):
            # subplot(1,num_stimulate,i)
            plt.subplot(int(f'{num_stimulate}1{i}'))
            sub_X = []
            sub_Y = []
            left  = stimulus - ceil(self.before*self.fps)
            right = stimulus + ceil(self.after *self.fps)
            for j in len(range(left, right)): # 取上界，包括[before, after]区间
                if j in self.frames:
                    sub_X.append(j)
                    sub_Y.append(data[self.frames.index(j)])
            plt.plot(sub_X, sub_Y, c=colors[0])
            plt.scatter(stimulus, data[self.frames.index(stimulus)])
            plt.title(name + f': stimulus {i}')
            plt.xlabel(xlabel)
            plt.xlim(left - 5, right + 5)
            plt.ylabel(ylabel)
            
        plt.show()
    
    """@depricated: segment by frames if it is in a section of stimulus, return list"""   
    def frame_segment_1(self):
        frame_sections = [] # 所有刺激包括的帧, 按照stimulate
        pf = 0
        flag = 0
        for i in range(self.lighttime):
            frame_sections.append([])
            while(True):
                if(self.in_section(pf, i)):
                    flag = 1
                    frame_sections[-1].append(pf)
                else:
                    if flag == 1:
                        flag = 0
                        pf = pf + 1
                        break
                pf = pf + 1
        return frame_sections
    
    """@depricated another form of frame_segment"""
    def frame_segment_2(self):
        frame_sections = [] # 所有刺激包括的帧, 按照stimulate
        pf = 0
        flag = 0
        for i in range(self.lighttime):
            frame_sections.append([])
            while True:
                if(flag):
                    if not self.in_section(pf, i):
                        flag = 0
                        break
                else:
                    if self.in_section(pf, i):
                        flag = 1
                        frame_sections.append(pf)
                pf = pf + 1
        return frame_sections
    
    """return every possible frame number in each segment of stimulus, return list"""
    def sti_segment(self) -> list:
        sti_sections = []
        for stimulus in self.lighttime:
            sti_sections.append([])
            left  = stimulus - ceil(self.before*self.fps)
            right = stimulus + ceil(self.after *self.fps)
            for f in range(left, right+1):
                if f in self.frames:
                    sti_sections[-1].append(self.frames.index(f))
        assert len(sti_sections) == len(self.lighttime), "Wrong Value"
        return sti_sections
        
    def showAngle(self,fps):
            
        """write file and plot simultaneously"""
        with open(f'results\Angle {self.filename},{self.timestr}.txt','w') as f:
            # plt.figure()
            # num_stimulate = len(self.lighttime)
            f.write('frame_num: angle(deg)')        
            for i, sti_ls in enumerate(self.stimus):
                # subplot(1,num_stimulate,i)
                # plt.subplot(int(f'{num_stimulate}1{i}'))
                plt.figure(f'pAngle-{i}')
                plt_x = []
                plt_y = []
                f.write(f'\nstimulus {i} ({len(sti_ls)} frames):\n')
                for pf in sti_ls:
                    x = self.frames[pf]
                    theta = self.Theta[pf]
                    if x in self.lighttime:
                        f.write(f'{x:3d}: {theta:.6f} (stimulate)\n')
                        plt.scatter(x,theta,c='y')
                    else:
                        f.write(f'{x:3d}: {theta:.6f}\n')
                    plt_x.append(x)
                    plt_y.append(theta)
                plt.plot(plt_x,plt_y,c='b')
                plt.xlabel('number of frame')
                plt.ylabel('angle(deg)')
                plt.title('angle curve')
                plt.savefig(f'fig\pAngle-stimulus{i}.png')
                plt.show()
                    
            # plt.show()
            f.write('end\n\nall frames: \n')
            for i in range(self.num):
                if i > 1 and self.frames[i] - self.frames[i-1] > 1:
                    f.write(black_sign+'\n')
                f.write(f'{self.frames[i]:3d}: {self.Theta[i]:.6f}\n')
            f.write('end\n')
        
        plt.figure('pAngle-interp')
        # plt.subplot(122)
        begin = 0
        end = len(self.frames)
        num = (end-begin)*10
        x_base = np.linspace(begin, end, num)
        y_curve = interpolate_b_spline(self.frames,self.Theta,x_base,der=0)
        y_der = interpolate_b_spline(self.frames,self.Theta,x_base,der=1)
        plt.plot(x_base,y_curve,c='g')
        # self.pAngle_interp.circle(x_base,y_curve,size=5,fill_color='green',fill_alpha=0.3)
        self.interp_omega = y_der
        self.interp_x = x_base
        
        """to be changed"""
        self.segment(-20,20)
        plt.xlabel('number of frame')
        plt.ylabel('angle(deg)')
        plt.title('interpolate angle curve')
        # output_file(filename="res_Angle.html", title="angle result")
        # save(self.pAngle)
        # save(self.pAngle_interp)
        points = [(self.frames[i],self.Theta[i]) for i in range(len(self.frames)) if i in self.stimus[0]]
        tList = np.linspace(0,1,200)
        # show_curve(points,tList)
        # show_der(points,tList)
        # show(self.pAngle)
        # show(self.pAngle_interp)
        plt.savefig('fig\pAngle_interp.png')
        plt.show()
        
    def showOmega(self,fps):
        self.adj = []
        for i in range(len(self.frames)):
            if i == 0:
                self.adj.append(0)
            elif self.frames[i] == self.frames[i-1]+1:
                self.adj.append(1)
            else:
                self.adj.append(0)
        
        plt.figure('pOmega')
        omega_center = 0
        omega_front = 0
        omega_back = 0
        omega_move = 0
        omega_min = pi/2
        omega_max = -pi/2
        flag = 0
        stimulate = 0
        if self.root:
            self.progressBar['maximum'] = self.lighttime[-1]
        for i in range(len(self.adj)):
            
            if self.root:
                self.progressBar['value'] = i
                self.root.update()
            
            if self.adj[i] and self.adj[i-1]:
                
                omega_center = (self.Theta[i] - self.Theta[i-1]) / fps
                omega_front = self.calc_1(i)
                omega_back = self.calc_2(i)
            
            elif self.adj[i]:
                
                omega_center = (self.Theta[i] - self.Theta[i-1]) / fps
                omega_front = 0
                omega_back = 0
                
            else:
                
                omega_center = 0
                omega_front = 0
                omega_back = 0
                
            omega_move = omega_front+omega_back-2*omega_center
            if omega_move > omega_max:
                omega_max = omega_move
            if omega_center > omega_max:
                omega_max = omega_center
            if omega_move < omega_min:
                omega_min = omega_move
            if omega_center < omega_min:
                omega_min = omega_center
            
            if stimulate==len(self.lighttime):
                break
            if flag == 0:
                if f2sec(i - self.lighttime[stimulate],fps) > -0.5:
                    flag = 1
                    plt.scatter(i,omega_center,c='b')
                    plt.scatter(i,omega_move,c='green')
                    # self.pOmega.circle(i,omega_center,size=10, line_color="white", fill_color="blue", fill_alpha=0.5)
                    # self.pOmega.circle(i,omega_move,size=10, line_color="white", fill_color="green", fill_alpha=0.5)
                    
            else:
                if f2sec(i - self.lighttime[stimulate],fps) > 4.5:
                    flag = 0
                    stimulate+=1
                elif abs(i - self.lighttime[stimulate]) < 2:
                    plt.scatter(i,omega_center,c='r')
                    plt.scatter(i,omega_move,c='#2D755C')
                    # self.pOmega.circle(i,omega_center,size=10, line_color="white", fill_color="red", fill_alpha=0.5)
                    # self.pOmega.circle(i,omega_move,size=10, line_color="white", fill_color="#2D755C", fill_alpha=0.5)
                    
                else:
                    plt.scatter(i,omega_center,c='b')
                    plt.scatter(i,omega_move,c='green')
                    # self.pOmega.circle(i,omega_center,size=10, line_color="white", fill_color="blue", fill_alpha=0.5)
                    # self.pOmega.circle(i,omega_move,size=10, line_color="white", fill_color="green", fill_alpha=0.5)
                    
        # self.pOmega.xaxis.axis_label = "帧序号"
        # self.pOmega.yaxis.axis_label = "转向角速度"
        plt.xlabel('number of frame')
        plt.ylabel('angular speed(deg/s)')
        plt.title('turning omega curve')
        self.segment(omega_min,omega_max)
        # output_file(filename="res_Angular_speed.html", title="angular speed result")
        # save(self.pOmega)
        # show(self.pOmega)
        plt.savefig('fig\pOmega.png')
        plt.show()
        
    def calc_1(self,i):
        k2 = atan((self.Y1[i]-self.Y1[i-1])/(self.X1[i]-self.X1[i-1])) if self.X1[i] != self.X1[i-1] else pi/2
        k1 = atan((self.Y1[i-1]-self.Y1[i-2])/(self.X1[i-1]-self.X1[i-2])) if self.X1[i-1] != self.X1[i-2] else pi/2
        return (k2 - k1) / self.fps if k2 != k1 else 0
        
    def calc_2(self,i):
        k2 = atan((self.Y2[i]-self.Y2[i-1])/(self.X2[i]-self.X2[i-1])) if self.X2[i] != self.X2[i-1] else pi/2
        k1 = atan((self.Y2[i-1]-self.Y2[i-2])/(self.X2[i-1]-self.X2[i-2])) if self.X2[i-1] != self.X2[i-2] else pi/2
        return (k2 - k1) / self.fps if k2 != k1 else 0
        
    def showPath(self,):
        plt.figure('pPath')
        print(len(self.frames))
        flag = 1 if len(self.X1) > 0 else 0 # if front and back path can be drawn
        for i,frame in enumerate(self.frames):
            if self.minDis(frame) < 1:
                print(frame)
                print(self.X_mid[i],self.Y_mid[i])
                plt.scatter(self.X_mid[i][1], self.Y_mid[i][1],c='r')
                if flag:
                    plt.scatter(self.X1[i], self.Y1[i], c='r')
                    plt.scatter(self.X2[i], self.Y2[i], c='r')
                
        plot_xmid = [i[1] for i in self.X_mid if self.in_range(i[0])]
        plot_ymid = [i[1] for i in self.Y_mid if self.in_range(i[0])]
        plt.plot(plot_xmid, plot_ymid,c='b')
        if flag:
            # print(self.stimus)
            # return
            plot_xf = [self.X1[i] for i in range(self.num) if self.frames[i] in self.stimus[0]]
            plot_yf = [self.Y1[i] for i in range(self.num) if self.frames[i] in self.stimus[0]]
            plot_xb = [self.X2[i] for i in range(self.num) if self.frames[i] in self.stimus[0]]
            plot_yb = [self.Y2[i] for i in range(self.num) if self.frames[i] in self.stimus[0]]
        
            plt.plot(plot_xf,plot_yf,c='green')
            plt.plot(plot_xb,plot_yb,c='purple')
        plt.xlabel(f'x({self.str_scale})')
        plt.ylabel(f'y({self.str_scale})')
        plt.title('path curve')
        plt.savefig('fig\pPath.png')
        plt.show()
        
    def showCurve(self,):
        self.radius = []
        max_r = 0
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        for i in range(len(self.X_mid) - 2):
            if self.X_mid[i+2][0]-self.X_mid[i+1][0] == 1 and self.X_mid[i+1][0]-self.X_mid[i][0] == 1: # 连续三点
                cnt1 += 1
                d_s = sqrt((self.Y_mid[i+1][1]-self.Y_mid[i][1])**2 +(self.X_mid[i+1][1]-self.X_mid[i][1])**2)
                d_thres = 0.001*self.out_ratio if self.out_ratio > 0 else 0.001
                if d_s > d_thres:
                    cnt2 += 1
                    alpha1 = (atan((self.Y_mid[i+2][1] - self.Y_mid[i+1][1]) / (self.X_mid[i+2][1] - self.X_mid[i+1][1])) 
                            if abs(self.X_mid[i+2][1] - self.X_mid[i+1][1]) > d_thres else pi/2)
                    alpha2 = (atan((self.Y_mid[i+1][1] - self.Y_mid[i][1]) / (self.X_mid[i+1][1] - self.X_mid[i][1])) 
                            if abs(self.X_mid[i+1][1] - self.X_mid[i][1]) > d_thres else pi/2)
                    d_alpha = alpha1 - alpha2
                    if d_alpha > 0.001:
                        cnt3 += 1
                        r = d_s / d_alpha
                        """changed: withdrew;"""
                        # if self.out_ratio:
                        #     r = r*self.out_ratio
                        self.radius.append(r)
                        if r > max_r:
                            max_r = r
                    else:
                        self.radius.append(0)
                else:
                    self.radius.append(0)
            else: # these three situation we can't calculate the radius, default 0
                self.radius.append(0)
        assert len(self.radius) == self.num - 2, ValueError("Wrong Value")
        print('filter:',self.num, cnt1, cnt2, cnt3)
                
        """write file and plot simultaneously"""
        with open(f'results\Turning Radius {self.filename},{self.timestr}.txt','w') as f:
            f.write(f'frame_num: radius({self.str_scale})')        
            for i, sti_ls in enumerate(self.stimus):
                plt.figure(f'pRadius-{i}')
                plt_x = []
                plt_y = []
                f.write(f'\nstimulus {i} ({len(sti_ls)} frames):\n')
                for pf in sti_ls:
                    x = self.frames[pf]
                    r = self.radius[pf]
                    if x in self.lighttime:
                        f.write(f'{x:3d}: {r:.6f} (stimulate)\n')
                        plt.scatter(x,r,c='r')
                    else:
                        f.write(f'{x:3d}: {r:.6f}\n')
                    if r > 0:
                        plt_x.append(x)
                        plt_y.append(r)
                    else:
                        if x in self.lighttime:
                            pass
                        else:
                            plt.scatter(x,0,c='y')
                # print(plt_x)
                plt.plot(plt_x,plt_y,c='b')
                plt.xlabel('number of frame')
                plt.ylabel(f'radius({self.str_scale})')
                plt.title('turning radius')
                plt.savefig(f'fig\pRadius-stimulus{i}.png')
                plt.show()
                    
            f.write('end\n\nall frames: \n')
            for i in range(self.num - 2):
                if i > 1 and self.frames[i] - self.frames[i-1] > 1:
                    f.write(black_sign+'\n')
                f.write(f'{self.frames[i]:3d}: {self.radius[i]:.6f}\n')
            f.write('end\n')
        return
        
    def showDist(self):
        pass

if pstatus == "debug":
    if __name__ == '__main__':
        data_dealer = Dealer()
        data_dealer.deal_time(open('out-light-every.txt','r'), 30)
        data_dealer.parse_fbpoints(open('out-meanshift-1.txt','r'),open('out-meanshift-2.txt','r'),30)
        data_dealer.data_change_ratio(0.012)
        data_dealer.To_centimeter(0.012)
        # data_dealer.parse_center_angle(open('out-contour-center.txt','r'),open('out-contour-theta.txt','r'),30)
        # data_dealer.showPath()
        data_dealer.showCurve()
        # data_dealer.showAngle(30)
        # data_dealer.showOmega(30)
    
class Cheker:
    def __init__(self,cap,status) -> None:
        self.cap = cap
        self.status = status
        
    def check(self,dealer):
        if self.status == None:
            showinfo(message='请先处理视频')
            return
        if self.status == 'contour':
            pass
        else:
            dealer.showDist()
            