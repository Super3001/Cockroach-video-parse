# deal_data.py
import numpy as np
from math import comb, ceil, pi, atan, sqrt
from tkinter.messagebox import showinfo, showwarning
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

fontdict_title = {'weight':'normal','size': 20}
fontdict_label = {'weight': 'normal', 'size': 13}

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

'''去除异常值'''
def remove_abnormal(points:np.array):
    std = points.std()
    median = np.median(points)
    for i in range(len(points)-1):
        if abs(points[i+1]-points[i]) > 6*std:
            if abs(points[i+1] - median) > abs(points[i] - median):
                points[i+1] = points[i]
            else:
                points[i] = points[i+1]
    
    return points
            
class Dealer(DataParser):
    def __init__(self,fps=60,filename=None,root=None,progressBar=None,markstr=None,skip_n=1,plot_tool='plt', light=True) -> None:
        super().__init__(fps=fps,skip_n=skip_n, light=light)
        self.root = root
        self.progressBar = progressBar

        self.filename = filename
        self.timestr = utils.timestr()
        self.out_ratio = 0
        self.str_scale = 'px'
        self.plot_tool = plot_tool
        self.stimulateLabelFlag = False

    def save_midpoint_data(self):
        with open(f'results\Pos {self.filename},{self.timestr}.txt','w') as f:
            f.write(f'frame# \tX_mid({self.str_scale}) \tY_mid({self.str_scale})\n')
            for i, frame in enumerate(self.frames):
                if self.str_scale == 'cm':
                    f.write(f'{frame:<3d} \t{self.X_mid[i]:.3f} \t\t{self.Y_mid[i]:.3f}\n')
                else:  
                    f.write(f'{frame:<3d} \t{self.X_mid[i]} \t\t{self.Y_mid[i]}\n')
            f.write('end\n')
        
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
            if f-i >= -0.5*self.fps and f-i <= self.after*self.fps:
                return True
        return False
    
    """origin line segment in one plot"""
    def segment(self,down,up):
        stimulate_label_flag = False
        for i in self.stimulus:
            x,y = [i-0.5*self.fps,i-0.5*self.fps],[down,up]
            if not stimulate_label_flag:
                plt.plot(x,y,color="red",label='stimulate start')
            else:
                plt.plot(x,y,color="red")
            x,y = [i+self.after*self.fps,i+self.after*self.fps],[down,up]
            if not stimulate_label_flag:
                plt.plot(x,y,color="navy",label='stimulate end')
                stimulate_label_flag = True
            else:
                plt.plot(x,y,color="navy")
    
    """@depricated: segment and plot in one function"""        
    """def segment_plt(self,data:list,xlabel,ylabel,name,colors=('b','y')):
        plt.figure()
        num_stimulate = len(self.stimulus)
        for i, stimulus in enumerate(self.stimulus):
            # subplot(1,num_stimulate,i)
            plt.subplot(int(f'{num_stimulate}1{i}'))
            sub_X = []
            sub_Y = []
            left  = stimulus - ceil(self.before*self.fps)
            right = stimulus + ceil(self.after *self.fps)
            for j in range(left, right): # 取上界，包括[before, after]区间
                if j in self.frames:
                    sub_X.append(j)
                    sub_Y.append(data[self.frames.index(j)])
            plt.plot(sub_X, sub_Y, c=colors[0])
            plt.scatter(stimulus, data[self.frames.index(stimulus)])
            plt.title(name + f': stimulus {i}')
            plt.xlabel(xlabel)
            plt.xlim(left - 5, right + 5)
            plt.ylabel(ylabel)
            
        plt.show()"""
    
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
        plt.plot(x,y,color='y')
    
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
            
    def showAngle(self):
        self.stimulateLabelFlag = False
        """write file and plot simultaneously"""
        with open(f'results\Angle {self.filename},{self.timestr}.txt','w') as f:
            # plt.figure()
            f.write('frame#\t angle(deg)\n')        
            # print(self.durings)
            # print(self.frames[self.durings[0]])
            if self.has_light:
                for i, sti_ls in enumerate(self.durings):
                    sti_i = np.argmin(abs(self.frames[sti_ls] - self.stimulus[i]))
                    # subplot(1,num_stimulate,i)
                    # plt.subplot(int(f'{num_stimulate}1{i}'))
                    plt.figure(f'pAngle-{i}')
                    plt_x = []
                    plt_y = []
                    f.write(f'=== stimulus {i} ({len(sti_ls)} frames) ===\n')
                    for j, pf in enumerate(sti_ls):
                        pf = int(pf)
                        x = self.frames[pf]
                        theta = self.Theta[pf]
                        # if x in self.stimulus:
                        if j == sti_i:
                            f.write(f'{x:<3d}\t {theta:.6f} (stimulate)\n')
                            if not self.stimulateLabelFlag:
                                plt.scatter(x,theta,c='r',label='stimulate')
                                self.stimulateLabelFlag = True
                            else:
                                plt.scatter(x,theta,c='r')
                        else:
                            f.write(f'{x:<3d}\t {theta:.6f}\n')
                        plt_x.append(x)
                        plt_y.append(theta)

                    plt_assist_f = self.assist_angle_f[sti_ls] if hasattr(self,'assist_angle_f') else None
                    plt_assist_b = self.assist_angle_b[sti_ls] if hasattr(self,'assist_angle_b') else None
                    
                    plt.plot(plt_x, plt_y, c='b', label='angle')
                    if plt_assist_f is not None:
                        plt.plot(plt_x, plt_assist_f, c=(0,0.9,0),label='assist-angle(front)') # c: (r,g,b)
                    if plt_assist_b is not None:
                        plt.plot(plt_x, plt_assist_b, c=(0.9,0,0.9),label='assist-angle(back)')
                    
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.title('Plotting Curve')
                    plt.xlabel('number of frame')
                    plt.ylabel('angle(deg)') # 单位：角度
                    plt.title(f'angular displacement {i+1}', fontdict_title)
                    plt.legend()
                    plt.savefig(f'fig\pAngle-stimulus{i+1}.png')
                    plt.show()
                        
                # plt.show()
                f.write('end\n\nall frames: \n')
            else:
                f.write('all frames:\n')
            for i in range(self.num):
                if i > 1 and self.frames[i] - self.frames[i-1] > 1:
                    f.write(black_sign+'\n')
                f.write(f'{self.frames[i]:<3d}\t {self.Theta[i]:.6f}\n')
            f.write('end\n')
        
        plt.figure('pAngle-interp')
        # plt.subplot(122)
        begin = 0
        end = np.max(self.frames) # 包括所有帧
        num = (end-begin)*10
        x_base = np.linspace(begin, end, num)
        y_curve = interpolate_b_spline(self.frames,self.Theta,x_base,der=0)
        y_der = interpolate_b_spline(self.frames,self.Theta,x_base,der=1)
        
        y_curve = remove_abnormal(y_curve)
        plt.plot(x_base,y_curve,c='g')
        # self.pAngle_interp.circle(x_base,y_curve,size=5,fill_color='green',fill_alpha=0.3)
        self.interp_omega = y_der
        self.interp_x = x_base
        
        """to be changed"""
        if self.has_light: self.segment(np.min(y_curve[10:])-10,np.max(y_curve[10:])+10)
        plt.xlabel('number of frame')
        plt.ylabel('angle(deg)')
        plt.title('interpolate angular displacement', fontdict_title)
        if self.has_light: plt.legend()
        plt.savefig('fig\pAngle_interp.png')
        plt.show()
        
    def showOmega(self):
        if not self.has_light:
            self.show_omega_all()
            return
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
        # if self.root:
        #     self.progressBar['maximum'] = round(self.stimulus[-1] / self.skip_n)
        
        max_f = 0
        for i in range(len(self.frames_adj)-1):
            f = self.frames[i]
            
            # if self.root:
            #     self.progressBar['value'] = i
            #     self.root.update()
            
            if self.frames_adj[i] and self.frames_adj[i+1]:
                
                omega_center = (self.Theta[i+1] - self.Theta[i]) / self.fps
                if move_flag: # 可以计算摆动角速度
                    omega_front = self.calc_1(i+1)
                    omega_back = self.calc_2(i+1)
            elif self.frames_adj[i]:
                omega_center = (self.Theta[i] - self.Theta[i-1]) / self.fps
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
                if f2sec(f - self.stimulus[stimulate],self.fps) > -0.5:
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
                if f2sec(f - self.stimulus[stimulate],self.fps) > self.after:
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

        # xh = [-1,max_f+1]; yh = [0, 0]
        # plt.plot(xh, yh, color='black')  # 绘制直线，设置颜色为...
        
        plt.xlabel('number of frame')
        plt.ylabel('angular velocity(deg/s)')
        plt.title('angular velocity', fontdict_title)
        self.segment(omega_min,omega_max)
        self.horizontal(0,max_f)
        # plt.legend(bbox_to_anchor=(1.05,0), loc=3, borderaxespad=0)
        plt.legend()
        # output_file(filename="res_Angular_velocity.html", title="angular velocity result")
        # save(self.pOmega)
        # show(self.pOmega)
        plt.savefig('fig\pOmega.png')
        plt.show()
        
    def show_omega_all(self):
        move_flag = True if len(self.X1) > 0 else False
        plt.figure('pOmega')
        omega_center = 0
        omega_front = 0
        omega_back = 0
        omega_move = 0
        omega_min = pi/2
        omega_max = -pi/2
        for i in range(len(self.frames_adj)-1):
                f = self.frames[i]
                
                if self.frames_adj[i] and self.frames_adj[i+1]:
                    omega_center = (self.Theta[i+1] - self.Theta[i]) / self.fps
                    if move_flag: # 可以计算摆动角速度
                        omega_front = self.calc_1(i+1)
                        omega_back = self.calc_2(i+1)
                elif self.frames_adj[i]:
                    omega_center = (self.Theta[i] - self.Theta[i-1]) / self.fps
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
                    
                if move_flag:
                    plt.scatter(f,omega_move,c='green')
                plt.scatter(f,omega_center,c='b')
                
        plt.xlabel('number of frame')
        plt.ylabel('angular velocity(deg/s)')
        plt.title('angular velocity', fontdict_title)
        self.nframe = max(self.frames)
        self.horizontal(-2,self.nframe+2)
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
        self.save_midpoint_data()
        if not self.has_light:
            self.show_path_all()
            return
        # show_path_light
        plt.figure('pPath')
        flag = 1 if len(self.X1) > 0 else 0 # if front and back path can be drawn
        
        cbar = plot.colorbar_between_two(length=len(self.stimulus))
        
        for i, f in enumerate(self.stimulus):
            """ 距离刺激帧最近的一帧 """
            idx = np.argmin(np.abs(self.frames - f))

            # ·刺激置于最上层,zorder=100·
            if not self.stimulateLabelFlag:
                plt.scatter(self.X_mid[idx], self.Y_mid[idx],color=tuple(cbar[i]),zorder=100, label='stimulate')
                self.stimulateLabelFlag = True
            else:
                plt.scatter(self.X_mid[idx], self.Y_mid[idx],color=tuple(cbar[i]),zorder=100)
            if flag:
                plt.scatter(self.X1[idx], self.Y1[idx], color=tuple(cbar[i]),zorder=100)
                plt.scatter(self.X2[idx], self.Y2[idx], color=tuple(cbar[i]),zorder=100)

        """ 画出所有在刺激范围内的点 """
        plot_xmid = [self.X_mid[i] for i in range(self.num) if self.in_range(self.frames[i])]
        plot_ymid = [self.Y_mid[i] for i in range(self.num) if self.in_range(self.frames[i])]
        plt.plot(plot_xmid, plot_ymid,c='b',label='mid')

        # 星星置于中间层
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
                        
        plt.xlabel(f'x({self.str_scale})', fontdict_label)
        plt.ylabel(f'y({self.str_scale})', fontdict_label)
        # plt.legend(bbox_to_anchor=(1.05,0), loc=3, borderaxespad=0)
        plt.legend()
        plt.title('path curve', fontdict=fontdict_title)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.13)
        plt.savefig('fig\pPath.png')
        plt.show()
        
        """ 分刺激画出对应的点 """
        if len(self.durings) > 1:
            for i, indice in enumerate(self.durings):
                if len(indice) == 0:
                    continue
                xmid = self.X_mid[indice]; ymid = self.Y_mid[indice]; plt.plot(xmid, ymid, c='b', label='mid')
                if flag: 
                    x1 = self.X1[indice]; 
                    y1 = self.Y1[indice]; 
                    x2 = self.X2[indice]; 
                    y2 = self.Y2[indice]
                    plt.plot(x1,y1,c='green',label='front'); plt.plot(x2,y2,c='purple',label='back')
                
                idx = np.argmin(np.abs(self.frames - self.stimulus[i])) # 标记刺激帧的下标
                # print(indice)
                # print(idx)
                
                plt.scatter(self.X_mid[idx], self.Y_mid[idx],color=(1,0,0))
                if flag:
                    plt.scatter(self.X1[idx], self.Y1[idx], color=(1,0,0))
                    plt.scatter(self.X2[idx], self.Y2[idx], color=(1,0,0))
                plt.xlabel(f'x({self.str_scale})', fontdict_label)
                plt.ylabel(f'y({self.str_scale})', fontdict_label)
                # plt.legend(bbox_to_anchor=(1.05,0), loc=3, borderaxespad=0)
                plt.legend()
                plt.title(f'path curve: stimulus{i+1}', fontdict_title)
                plt.savefig(f'fig\pPath_{i+1}.png')
                plt.show()
                
    def show_path_all(self):
        plt.figure('pPath')
        flag = 1 if len(self.X1) > 0 else 0 # if front and back path can be drawn
        
        plt.plot(self.X_mid, self.Y_mid,c='b',label='mid')
        plt.scatter(self.X_mid[0], self.Y_mid[0], color='#FA9A3F', s=50, marker='*', label='start', zorder=50)
        plt.scatter(self.X_mid[-1], self.Y_mid[-1], color='#49DBF5', s=50, marker='*', label='end', zorder=50)
        
        if flag:
            plt.plot(self.X1,self.Y1,c='green',label='front')
            plt.plot(self.X2,self.Y2,c='purple',label='back')
            plt.scatter(self.X1[0], self.Y1[0], color='#FA9A3F', s=50, marker='*', zorder=50)
            plt.scatter(self.X2[0], self.Y2[0], color='#FA9A3F', s=50, marker='*', zorder=50)
            plt.scatter(self.X1[-1], self.Y1[-1], color='#49DBF5', s=50, marker='*', zorder=50)
            plt.scatter(self.X2[-1], self.Y2[-1], color='#49DBF5', s=50, marker='*', zorder=50)
                            
        plt.xlabel(f'x({self.str_scale})', fontdict=fontdict_label)
        plt.ylabel(f'y({self.str_scale})', fontdict=fontdict_label)
        plt.legend()
        plt.title('path curve', fontdict_title)
        plt.savefig('fig\pPath.png')
        plt.show()
    
    def gen_curve_points(self):
        """this function is to select unique points of path
        """
        curve_point = []
        cp_indexes = []
        for i in range(self.num):
            point = [round(float(self.X_mid[i]),1), round(float(self.Y_mid[i]),1)] # 精度为 0.1px
            if point not in curve_point:
                curve_point.append(point)
                cp_indexes.append([i])
            else:
                cp_indexes[-1].append(i)
        
        self.curve_point = curve_point
        self.cp_indexes = cp_indexes

    def showCurve(self):
        if self.has_light == 0:
            return self.show_curve_all()
        with open(f'results\Curvature {self.filename},{self.timestr}.txt','w') as f:

            _type = 'mean'
            route_length = 0
            for i in range(self.num-1):
                dy = self.Y_mid[i+1] - self.Y_mid[i]
                dx = self.X_mid[i+1] - self.X_mid[i]
                route_length += sqrt(dx**2 + dy**2)
            
            route_angle = 0
            for i in range(self.num-1):
                d_theta = abs(self.Theta[i+1] - self.Theta[i])
                route_angle += d_theta
                
            mean_radius = route_length / route_angle
            mean_curv = 1 / mean_radius
            
            i_sti = 0
            curvs = []
            for i, sti_ls in enumerate(self.durings):
                xmid_sti = self.X_mid[sti_ls]
                ymid_sti = self.Y_mid[sti_ls]
                # 计算轨迹长度
                route_length = np.sqrt((xmid_sti[1:]-xmid_sti[:-1])**2 + (ymid_sti[1:]-ymid_sti[:-1])**2).sum()
                # 计算转向角度
                theta_sti = self.Theta[sti_ls]
                route_angle = np.abs(theta_sti[1:] - theta_sti[:-1]).sum()
                # route_angle = np.abs(self.Theta[sti_ls[1:]] - self.Theta[sti_ls[:-1]]).sum()
                # 计算平均曲率
                mean_curv = route_angle / route_length
                
                curvs.append(mean_curv)
                
            f.write('mean curvature of all stimulus:\n')
            f.write(f'stimulus#\t curvature({self.str_scale}^-1)\n')
            for i in range(len(curvs)):
                f.write(f'{i+1:<3d}\t\t {curvs[i]:.4f}\n')
            f.write('end\n')
                
            plt.figure()
            plt.bar(np.arange(len(curvs)), curvs)
            # 在每个柱状图顶端标上值
            for i in range(len(curvs)):
                plt.text(i, curvs[i], str(round(curvs[i],2)), ha='center', va='bottom')
            plt.xlabel('number of stimulus')
            plt.ylabel(f'curvature({self.str_scale})^-1')
            plt.xlim(-1, len(curvs))
            plt.title(f'curvature', fontdict=fontdict_title)     
            plt.savefig(f'fig\pCurve-sti.png')
            plt.show()
                
            self.curvature = np.array(curvs)
                
            _type = 'arc'
            self.curvature = np.zeros(self.num)
            max_curv = 0
            self.gen_curve_points()

            for i in range(1, len(self.curve_point)-1):
                r = self.radius_arc_of_points(*(self.curve_point[i-1:i+2]))
                if r == np.inf or r == 0:
                    curv = 0
                else:
                    curv = 1 / r
                if curv > max_curv:
                    max_curv = curv
                self.curvature[self.cp_indexes[i]] = curv

            """画出所有点的curvature（目前取消此步骤）"""
            # plt.figure('pCurve')
            # plt_x = []
            # plt_y = []
            # for i, r in enumerate(self.curvature):
            #     if r == 0:
            #         # plt.scatter(self.frames[i],0,c='y')
            #         pass
            #     else:
            #         plt_x.append(self.frames[i])
            #         plt_y.append(r)
            # plt.plot(plt_x, plt_y, c='b')
            # plt.scatter(self.stimulus, np.zeros(len(self.stimulus)), c='r')

            # # 画一条y=0
            # xh = [-1,self.frames[-1]+1]; yh = [0, 0]
            # plt.plot(xh, yh, color='black')  # 绘制直线，设置颜色为红色

            # plt.xlabel('number of frame')
            # plt.ylabel(f'curvature({self.str_scale})')
            # plt.title(f'curvature')
            # plt.savefig(f'fig\pCurve.png')
            # plt.show()

            # plt.figure('subpCurve')
            
            f.write('\nall frames: \n')
            for i in range(self.num):
                # if i > 1 and self.frames[i] - self.frames[i-1] > 1:
                #     f.write(f'{self.frames[i]:3d}: ----- black -----\n')
                f.write(f'{self.frames[i]:<3d}\t\t {self.curvature[i]:.4f}\n')
            f.write('end\n')

    def show_curve_all(self):
        
        """calc curvature"""
        _type = 'arc'
        self.curvature = np.zeros(self.num)
        max_curv = 0
        self.gen_curve_points()

        for i in range(1, len(self.curve_point)-1):
            r = self.radius_arc_of_points(*(self.curve_point[i-1:i+2]))
            if r == np.inf or r == 0:
                curv = 0
            else:
                curv = 1 / r
            if curv > max_curv:
                max_curv = curv
            self.curvature[self.cp_indexes[i]] = curv
        
        with open(f'results\Curvature {self.filename},{self.timestr}.txt','w') as f:
            f.write(f'stimulus#\t curvature({self.str_scale})\n')
            for i in range(self.num):
                f.write(f'{self.frames[i]:<3d}\t\t {self.curvature[i]:.4f}\n')
            f.write('end\n')
        showinfo('','数据已保存，未设置图像')

    def radius_arc_of_points(self, A, B, C):
        x1, y1 = A
        x2, y2 = B
        x3, y3 = C
        if np.all(A == B) or np.all(B == C) or np.all(A == C):
            return 0
        if (y2 - y1)*(x3 - x1) == (y3 - y1)*(x2 - x1): # 共线
            return np.inf 
        
        # 防止k(中垂线的斜率)为正无穷
        if y1 == y2:
            # 交换B和C
            x2, x3 = x3, x2
            y2, y3 = y3, y2
        elif y2 == y3:
            # 交换A和B
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        # 计算中点M和N
        M = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        N = np.array([(x2 + x3) / 2, (y2 + y3) / 2])

        # 计算斜率k1和k2
        # k1 = -1 / ((y2 - y1) / (x2 - x1))
        # k2 = -1 / ((y3 - y2) / (x3 - x2))
        k1 = (x1 - x2) / (y2 - y1)
        k2 = (x2 - x3) / (y3 - y2)

        # 计算直线L1和L2的截距b1和b2
        b1 = M[1] - k1 * M[0]
        b2 = N[1] - k2 * N[0]

        # 计算圆心的坐标
        x_center = (b2 - b1) / (k1 - k2)
        y_center = k1 * x_center + b1

        # 计算圆的半径
        radius = np.sqrt((x1 - x_center)**2 + (y1 - y_center)**2)

        return radius

    '''旧版showCurve'''
    '''def showCurve(self,):
        self.radius = []
        max_r = 0
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        # stdoutpb = utils.Stdout_progressbar(self.num-2)
        # stdoutpb.reset()
        for i in range(self.num - 2):
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
        with open(f'results\Curvature {self.filename},{self.timestr}.txt','w') as f:
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
                    plt.scatter(self.stimulus[i],0,c='r',zorder=100)
                    
                plt.plot(plt_x,plt_y,c='b')
                plt.xlabel('number of frame')
                plt.ylabel(f'radius({self.str_scale})')
                plt.title('Curvature')
                plt.savefig(f'fig\pRadius-stimulus{i+1}.png')
                plt.show()
                    
            f.write('end\n\nall frames: \n')
            for i in range(self.num - 2):
                if i > 1 and self.frames[i] - self.frames[i-1] > 1:
                    f.write(f'{self.frames[i]:3d}: ----- black -----\n')
                f.write(f'{self.frames[i]:3d}: {self.radius[i]:.6f}\n')
            f.write('end\n')
        return'''
        
    def showDist(self):
        '''
        show how the dist between f and b varies
        '''
        if len(self.D) == 0:
            showwarning('warning', '未提取前后点，无法用该种方法估算精度')
            return
        plt.figure()
        plt.plot(self.frames, self.D)
        plt.xlabel('number of frame')
        plt.ylabel(f'distance unit:{self.str_scale}')
        plt.title('distance', fontdict=fontdict_title)
        plt.savefig('fig\pDist.png')
        plt.show()
        
        with open(f'results\Stability {self.filename},{self.timestr}.txt','w') as f:
            f.write(f'frame#\t dist({self.str_scale})\n')
            for i, each in enumerate(self.D):
                f.write(f'{self.frames[i]:<3d}\t {each:.2f}\n')
            f.write('end\n')
        
        return

    def showRadius(self):
        pass

if pstatus == "debug":
    if __name__ == '__main__':
        data_dealer = Dealer(30)
        data_dealer.parse_light(open('out-light-every.txt','r'), 30)
        data_dealer.parse_fbpoints(open('out-color-1.txt','r'),open('out-color-2.txt','r'),30)
        # data_dealer.parse_center_angle(open('out-contour-center.txt','r'),open('out-contour-theta.txt','r'),30)
        # data_dealer.parse_feature_result(open('out-feature-1.txt','r'),open('out-feature-2.txt','r'),30)
        # data_dealer.data_change_ratio(0.012)
        # data_dealer.To_centimeter(0.012)
        data_dealer.showPath()
        # data_dealer.save_midpoint_data()
        # data_dealer.showCurve()
        # data_dealer.showAngle()
        # data_dealer.showOmega()
        # data_dealer.showDist()
    