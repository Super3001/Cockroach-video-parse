# deal_data.py

import numpy as np
from math import comb, ceil, pi, sqrt, atan
from tkinter.messagebox import showinfo
from scipy import interpolate
import matplotlib.pyplot as plt
import utils
import plot
from parse_data import DataParser, f2sec
from signals import black_sign
import matplotlib
matplotlib.use('TkAgg')

"""debug global property"""
from control import pstatus
# pstatus = "release"
# pstatus = "debug"

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

class Dealer(DataParser):
    def __init__(self,fps=60,filename=None,root=None,progressBar=None,markstr=None,skip_n=1,plot_tool='plt') -> None:
        super().__init__(fps=fps,skip_n=skip_n)
        self.root = root
        self.progressBar = progressBar

        self.filename = filename
        self.timestr = utils.timestr()
        self.out_ratio = 0
        self.str_scale = 'px'
        self.plot_tool = plot_tool

    def To_centimeter(self, ratio):
        self.out_ratio = ratio
        self.str_scale = 'cm'
        
    def To_origin(self):
        self.out_ratio = 0
        self.str_scale = 'px'

    def minDis(self,f):
        min = 1e6
        for i in self.stimulus:
            if abs(f - i)<min:
                min = abs(f-i)
        return min
    
    """pf: self.frames[pf], pt: self.stimulus[pt]"""
    def in_section(self, pf, pt):
        frame = self.frames[pf]
        stimulus = self.stimulus[pt]
        left  = stimulus - ceil(self.before*self.fps)
        right = stimulus + ceil(self.after *self.fps)
        if frame >= left and frame <= right:
            return True
        return False
        
    """@depricated: judge if a frame number in section of stimulate"""
    def in_range(self,f):
        for i in self.stimulus:
            if f-i >= -0.5*self.fps and f-i <= 4.5*self.fps:
                return True
        return False
    
    """origin line segment in one plot"""
    def segment(self,down,up):
        for i in self.stimulus:
            x,y = [i-0.5*self.fps,i-0.5*self.fps],[down,up]
            plt.plot(x,y,color="navy")
            x,y = [i+4.5*self.fps,i+4.5*self.fps],[down,up]
            plt.plot(x,y,color="red")
    
    """@depricated: segment and plot in one function"""        
    def segment_plt(self,data:list,xlabel,ylabel,name,colors=['b','y']):
        plt.figure()
        num_stimulate = len(self.stimulus)
        for i, stimulus in enumerate(self.stimulus):
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
        for i in range(self.stimulus):
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
    
    def horizontal(self,left,right,value=0):
        x,y = [left, right],[value,value]
        plt.plot(x,y,color='black')
    
    """@depricated another form of frame_segment"""
    def frame_segment_2(self):
        frame_sections = [] # 所有刺激包括的帧, 按照stimulate
        pf = 0
        flag = 0
        for i in range(self.stimulus):
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
            
    def showAngle(self,fps):
            
        """write file and plot simultaneously"""
        with open(f'results\Angle {self.filename},{self.timestr}.txt','w') as f:
            # plt.figure()
            # num_stimulate = len(self.stimulus)
            f.write('frame_num: angle(deg)')        
            # print(self.durings)
            # print(self.frames[self.durings[0]])
            for i, sti_ls in enumerate(self.durings):
                sti_i = np.argmin(abs(self.frames[sti_ls] - self.stimulus[i]))
                # subplot(1,num_stimulate,i)
                # plt.subplot(int(f'{num_stimulate}1{i}'))
                plt.figure(f'pAngle-{i}')
                plt_x = []
                plt_y = []
                f.write(f'\nstimulus {i} ({len(sti_ls)} frames):\n')
                for j, pf in enumerate(sti_ls):
                    pf = int(pf)
                    x = self.frames[pf]
                    theta = self.Theta[pf]
                    # if x in self.stimulus:
                    if j == sti_i:
                        f.write(f'{x:3d}: {theta:.6f} (stimulate)\n')
                        plt.scatter(x,theta,c='r')
                    else:
                        f.write(f'{x:3d}: {theta:.6f}\n')
                    plt_x.append(x)
                    plt_y.append(theta)
                '''change: 不连续点'''
                # plt.plot(plt_x,plt_y,c='b')
                for i in range(1, len(plt_x)):
                    if abs(plt_y[i] - plt_y[i-1]) > 90:
                        # plt.plot(x[i], y[i], 'o', c='r')  # 绘制跳变点
                        pass
                    else:
                        plt.plot([plt_x[i-1], plt_x[i]], [plt_y[i-1], plt_y[i]], '-', c='b')

                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Plotting Curve')
                plt.xlabel('number of frame')
                plt.ylabel('angle(deg)') # 单位：角度
                plt.title('angle curve')
                plt.savefig(f'fig\pAngle-stimulus{i+1}.png')
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
        end = np.max(self.frames) # 包括所有帧
        num = (end-begin)*10
        x_base = np.linspace(begin, end, num)
        y_curve = interpolate_b_spline(self.frames,self.Theta,x_base,der=0)
        y_der = interpolate_b_spline(self.frames,self.Theta,x_base,der=1)
        plt.plot(x_base,y_curve,c='g')
        # self.pAngle_interp.circle(x_base,y_curve,size=5,fill_color='green',fill_alpha=0.3)
        self.interp_omega = y_der
        self.interp_x = x_base
        
        """to be changed"""
        self.segment(np.min(y_curve)-10,np.max(y_curve)+10)
        plt.xlabel('number of frame')
        plt.ylabel('angle(deg)')
        plt.title('interpolate angle curve')
        # output_file(filename="res_Angle.html", title="angle result")
        # save(self.pAngle)
        # save(self.pAngle_interp)
        points = [(self.frames[i],self.Theta[i]) for i in range(len(self.frames)) if i in self.durings[0]]
        tList = np.linspace(0,1,200)
        # show_curve(points,tList)
        # show_der(points,tList)
        # show(self.pAngle)
        # show(self.pAngle_interp)
        plt.savefig('fig\pAngle_interp.png')
        plt.show()
        
    def showOmega(self,fps):
        # self.adj = []
        # for i in range(len(self.frames)):
        #     if i == 0:
        #         self.adj.append(0)
        #     elif self.frames[i] == self.frames[i-1]+1:
        #         self.adj.append(1)
        #     else:
        #         self.adj.append(0)
        move_flag = True if len(self.X1) > 0 else False
        plt.figure('pOmega')
        omega_center = 0
        omega_front = 0
        omega_back = 0
        omega_move = 0
        omega_min = pi/2
        omega_max = -pi/2
        flag = 0
        stimulate = 0 # 滑动计数方式
        i_stimulus = np.argmin(np.abs(self.frames - self.stimulus[stimulate]))
        # print(i_stimulus)
        # print(self.stimulus[stimulate])
        # print(self.frames)
        # exit(0)
        if self.root:
            self.progressBar['maximum'] = round(self.stimulus[-1] / self.skip_n)
        
        max_f = 0
        for i in range(len(self.frames_adj)-1):
            f = self.frames[i]
            
            if self.root:
                self.progressBar['value'] = i
                self.root.update()
            
            if self.frames_adj[i] and self.frames_adj[i+1]:
                
                omega_center = (self.Theta[i+1] - self.Theta[i]) / fps
                if move_flag: # 可以计算摆动角速度
                    omega_front = self.calc_1(i+1)
                    omega_back = self.calc_2(i+1)
            elif self.frames_adj[i]:
                omega_center = (self.Theta[i] - self.Theta[i-1]) / fps
                if move_flag:
                    omega_front = 0
                    omega_back = 0
            else:
                omega_center = 0
                if move_flag:
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
            
            if i == i_stimulus:
                if stimulate == 0:
                    if move_flag:
                        plt.scatter(f,omega_move,c='#4AC298',label=f'stimulus frame',zorder=100)
                    plt.scatter(f,omega_center,c='r',label=f'stimulus frame',zorder=100)
                    
                else:
                    if move_flag:
                        plt.scatter(f,omega_move,c='#4AC298',zorder=100)
                    plt.scatter(f,omega_center,c='r',zorder=100)
                # self.pOmega.circle(i,omega_center,size=10, line_color="white", fill_color="red", fill_alpha=0.5)
                # self.pOmega.circle(i,omega_move,size=10, line_color="white", fill_color="#4AC298", fill_alpha=0.5)
            
            if flag == 0:
                if f2sec(f - self.stimulus[stimulate],fps) > -0.5:
                    flag = 1
                    if stimulate == 0:
                        if move_flag:
                            plt.scatter(f,omega_move,c='green',label='omega_move')
                        plt.scatter(f,omega_center,c='b',label='omega_center')
                    else:
                        if move_flag:
                            plt.scatter(f,omega_move,c='green')
                        plt.scatter(f,omega_center,c='b')
                    # self.pOmega.circle(i,omega_center,size=10, line_color="white", fill_color="blue", fill_alpha=0.5)
                    # self.pOmega.circle(i,omega_move,size=10, line_color="white", fill_color="green", fill_alpha=0.5)
                    
            else:
                if f2sec(f - self.stimulus[stimulate],fps) > 4.5:
                    flag = 0
                    stimulate+=1
                    if stimulate==len(self.stimulus):
                        max_f = f
                        break
                    i_stimulus = np.argmin(np.abs(self.frames - self.stimulus[stimulate]))
                    # print(i_stimulus)
                    
                else:
                    if move_flag:
                        plt.scatter(f,omega_move,c='green')
                    plt.scatter(f,omega_center,c='b')
                    # self.pOmega.circle(i,omega_center,size=10, line_color="white", fill_color="blue", fill_alpha=0.5)
                    # self.pOmega.circle(i,omega_move,size=10, line_color="white", fill_color="green", fill_alpha=0.5)
                    
        # self.pOmega.xaxis.axis_label = "帧序号"
        # self.pOmega.yaxis.axis_label = "转向角速度"
        plt.xlabel('number of frame')
        plt.ylabel('angular speed(deg/s)')
        plt.title('turning omega curve')
        self.segment(omega_min,omega_max)
        self.horizontal(0,max_f)
        # plt.legend(bbox_to_anchor=(1.05,0), loc=3, borderaxespad=0)
        plt.legend()
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
        # print(len(self.frames))
        flag = 1 if len(self.X1) > 0 else 0 # if front and back path can be drawn

        
        cbar = plot.colorbar_between_two(length=len(self.stimulus))
        # print(cbar)
        # print('self.frames',self.frames)
        # print('self.stimulus',self.stimulus)
        # print('self.durings',self.durings)
        
        for i, f in enumerate(self.stimulus):
            """ 距离刺激帧最近的一帧 """
            idx = np.argmin(np.abs(self.frames - f))

            plt.scatter(self.X_mid[idx], self.Y_mid[idx],color=tuple(cbar[i]),zorder=100)
            if flag:
                plt.scatter(self.X1[idx], self.Y1[idx], color=tuple(cbar[i]),zorder=100)
                plt.scatter(self.X2[idx], self.Y2[idx], color=tuple(cbar[i]),zorder=100)

        # for i,frame in enumerate(self.frames):
        #     if self.minDis(frame) < 1:
        #         print(frame)
        #         print(self.X_mid[i],self.Y_mid[i])
        #         plt.scatter(self.X_mid[i], self.Y_mid[i],c='r')
        #         if flag:
        #             plt.scatter(self.X1[i], self.Y1[i], c='r')
        #             plt.scatter(self.X2[i], self.Y2[i], c='r')
                
        # plot_xmid = [i[1] for i in self.X_mid if self.in_range(i[0])]
        # print(self.durings); return
        
        """ 画出所有在刺激范围内的点 """
        plot_xmid = [self.X_mid[i] for i in range(self.num) if self.in_range(self.frames[i])]
        plot_ymid = [self.Y_mid[i] for i in range(self.num) if self.in_range(self.frames[i])]
        plt.plot(plot_xmid, plot_ymid,c='b',label='mid')
        plt.scatter(plot_xmid[0], plot_ymid[0], color='#FA9A3F', s=50, marker='*', label='start', zorder=50)
        plt.scatter(plot_xmid[-1], plot_ymid[-1], color='#49DBF5', s=50, marker='*', label='end', zorder=50)
        if flag:
            plot_xf = [self.X1[i] for i in range(self.num) if self.in_range(self.frames[i])]
            plot_yf = [self.Y1[i] for i in range(self.num) if self.in_range(self.frames[i])]
            plot_xb = [self.X2[i] for i in range(self.num) if self.in_range(self.frames[i])]
            plot_yb = [self.Y2[i] for i in range(self.num) if self.in_range(self.frames[i])]
        
            plt.plot(plot_xf,plot_yf,c='green',label='front')
            plt.plot(plot_xb,plot_yb,c='purple',label='back')
            plt.scatter(plot_xf[0], plot_yf[0], color='#FA9A3F', s=50, marker='*', zorder=50)
            plt.scatter(plot_xb[0], plot_yb[0], color='#FA9A3F', s=50, marker='*', zorder=50)
            plt.scatter(plot_xf[-1], plot_yf[-1], color='#49DBF5', s=50, marker='*', zorder=50)
            plt.scatter(plot_xb[-1], plot_yb[-1], color='#49DBF5', s=50, marker='*', zorder=50)
                        
        plt.xlabel(f'x({self.str_scale})')
        plt.ylabel(f'y({self.str_scale})')
        # plt.legend(bbox_to_anchor=(1.05,0), loc=3, borderaxespad=0)
        plt.legend()
        plt.title('path curve')
        plt.savefig('fig\pPath.png')
        plt.show()
        
        """ 分刺激画出对应的点 """
        if len(self.durings) > 1:
            for i, indice in enumerate(self.durings):
                if len(indice) == 0:
                    continue
                xmid = self.X_mid[indice]; ymid = self.Y_mid[indice]; plt.plot(xmid, ymid, c='b', label='mid')
                if flag: x1 = self.X1[indice]; y1 = self.Y1[indice]; x2 = self.X2[indice]; y2 = self.Y2[indice]; plt.plot(x1,y1,c='green',label='front'); plt.plot(x2,y2,c='purple',label='back')
                
                idx = np.argmin(np.abs(self.frames - self.stimulus[i])) # 标记刺激帧的下标
                # print(indice)
                # print(idx)
                
                plt.scatter(self.X_mid[idx], self.Y_mid[idx],color=(1,0,0))
                if flag:
                    plt.scatter(self.X1[idx], self.Y1[idx], color=(1,0,0))
                    plt.scatter(self.X2[idx], self.Y2[idx], color=(1,0,0))
                plt.xlabel(f'x({self.str_scale})')
                plt.ylabel(f'y({self.str_scale})')
                # plt.legend(bbox_to_anchor=(1.05,0), loc=3, borderaxespad=0)
                plt.legend()
                plt.title(f'path curve: stimulus{i+1}')
                plt.savefig(f'fig\pPath_{i+1}.png')
                plt.show()
        
    def showCurve(self,):
        self.radius = []
        max_r = 0
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        # stdoutpb = utils.Stdout_progressbar(self.num-2)
        # stdoutpb.reset()
        for i in range(len(self.X_mid) - 2):
            if self.frames_adj[i+1] and self.frames_adj[i]: # 连续三点
            # if self.X_mid[i+2][0]-self.X_mid[i+1][0] == 1 and self.X_mid[i+1][0]-self.X_mid[i][0] == 1: # 连续三点
                cnt1 += 1
                d_s = sqrt((self.Y_mid[i+1]-self.Y_mid[i])**2 +(self.X_mid[i+1]-self.X_mid[i])**2)
                d_thres = 0.001*self.out_ratio if self.out_ratio > 0 else 0.001
                if d_s > d_thres:
                    cnt2 += 1
                    alpha1 = (atan((self.Y_mid[i+2] - self.Y_mid[i+1]) / (self.X_mid[i+2] - self.X_mid[i+1])) 
                            if abs(self.X_mid[i+2] - self.X_mid[i+1]) > d_thres else pi/2)
                    alpha2 = (atan((self.Y_mid[i+1] - self.Y_mid[i]) / (self.X_mid[i+1] - self.X_mid[i])) 
                            if abs(self.X_mid[i+1] - self.X_mid[i]) > d_thres else pi/2)
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
            # stdoutpb.update(i+1)
        # stdoutpb.update(-1)
            
        assert len(self.radius) == self.num - 2, ValueError("Wrong Value")
        print('filter:',self.num, cnt1, cnt2, cnt3)
                
        # print(self.durings)
        """write file and plot simultaneously"""
        with open(f'results\Turning Radius {self.filename},{self.timestr}.txt','w') as f:
            f.write(f'frame_num: radius({self.str_scale})')        
            for i, sti_ls in enumerate(self.durings):
                plt.figure(f'pRadius-{i}')
                plt_x = []
                plt_y = []
                sti_flag = 0
                f.write(f'\nstimulus {i} ({len(sti_ls)} frames):\n')
                for pf in sti_ls: # pf代表下标idx
                    pf = int(pf)
                    if pf >= len(self.radius): # 最后两帧不算
                        break
                    x = self.frames[pf]
                    r = self.radius[pf]
                    if x in self.stimulus:
                        f.write(f'{x:3d}: {r:.6f} (stimulate)\n')
                        plt.scatter(x,r,c='r',zorder=100) # 刺激标志放在最上层
                        sti_flag = 1
                    else:
                        f.write(f'{x:3d}: {r:.6f}\n')
                    if r > 0:
                        plt_x.append(x)
                        plt_y.append(r)
                    else:
                        if x in self.stimulus:
                            pass
                        else:
                            plt.scatter(x,0,c='y')
                if sti_flag == 0:
                    plt.scatter(self.stimulus[i],0,c='r',zorder=100) # 解决跳读没有刺激标志的问题
                    
                # print(plt_x)
                plt.plot(plt_x,plt_y,c='b')
                plt.xlabel('number of frame')
                plt.ylabel(f'radius({self.str_scale})')
                plt.title('turning radius')
                plt.savefig(f'fig\pRadius-stimulus{i+1}.png')
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
        data_dealer = Dealer(60, skip_n=1)
        data_dealer.parse_light(open('out-light-every.txt','r'), 60)
        # data_dealer.parse_fbpoints(open('out-meanshift-1.txt','r'),open('out-meanshift-2.txt','r'),30)
        # data_dealer.parse_fbpoints(open('out-feature-1.txt','r'),open('out-feature-2.txt','r'),60)
        # data_dealer.data_change_ratio(0.012)
        # data_dealer.To_centimeter(0.012)
        data_dealer.parse_center_angle(open('out-contour-center.txt','r'),open('out-contour-theta.txt','r'),60)
        # data_dealer.showPath()
        # data_dealer.showCurve()
        # data_dealer.showAngle(30)
        data_dealer.showOmega(60)
    
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
            
  
