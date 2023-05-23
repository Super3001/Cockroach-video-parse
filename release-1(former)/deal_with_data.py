import numpy as np
from math import *
from tkinter.messagebox import *
from scipy import interpolate
import matplotlib.pyplot as plt

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
    
    def __init__(self,cap=None,root=None,progressBar=None) -> None:
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
        self.D = []
        self.X_mid = []
        self.Y_mid = []
        self.Theta=[]
        self.lighttime = []
        self.frames = []

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
            self.X_mid.append((i,xmid))
            self.Y_mid.append((i,ymid))
            dist=sqrt((self.X2[i]-self.X1[i])*(self.X2[i]-self.X1[i]) + (self.Y2[i]-self.Y1[i])*(self.Y2[i]-self.Y1[i]))
            if self.X2[i] - self.X1[i]==0:
                k=0
            else:
                k=(self.Y2[i]-self.Y1[i])/(self.X2[i]-self.X1[i])
            self.D.append((i,dist))
            self.K.append((i,k))
            self.Theta.append(atan(k)*180/pi)
        
        self.frames = [i[0] for i in self.X_mid]
        self.num = len(self.frames)
        self.stimus = [i for i in range(self.num) if self.in_range(i)]
        # print(self.Theta)
                    
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
            self.X_mid.append((cnt,x))
            self.Y_mid.append((cnt,y))
            cnt+=1
            
        self.Theta = []
        cnt = 0
        for i in data2:
            theta = float(i)
            self.Theta.append(theta)
            cnt += 1
        # showinfo(message='共检测到数据'+str(len(self.X_mid)))
        self.frames = [i[0] for i in self.X_mid]
            
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
        
        # print(len(self.lighttime))

    def minDis(self,f):
        min = 1e6
        for i in self.lighttime:
            if abs(f - i)<min:
                min = abs(f-i)
        return min
    
    def in_range(self,f):
        for i in self.lighttime:
            if f-i >= -0.5*self.fps and f-i <= 4.5*self.fps:
                return True
        return False
    
    def segment(self,down,up):
        for i in self.lighttime:
            x,y = [i-0.5*self.fps,i-0.5*self.fps],[down,up]
            plt.plot(x,y,color="navy")
            x,y = [i+4.5*self.fps,i+4.5*self.fps],[down,up]
            plt.plot(x,y,color="red")

    def showAngle(self,fps):
        plt.figure(0)      
        # plt.subplot(121)
        # self.pAngle = figure(width=400, height=400)
        # self.pAngle_interp = figure(width=600, height=600)
        # self.pAngle_bessel = figure(width=600, height=600)
        stimulate = 0
        self.stimus = []
        flag = 0
        # print(self.lighttime)
        for cnt in range(len(self.frames)):
            i = self.frames[cnt]
            k = self.Theta[cnt]
            if stimulate==len(self.lighttime):
                break
            if flag == 0:
                if f2sec(i - self.lighttime[stimulate],fps) > -0.5:
                    flag = 1
                    plt.scatter(i,k,c='b')
                    # self.pAngle.circle(i,k,size=10, line_color="white", fill_color="blue", fill_alpha=0.5)
                    self.stimus.append(cnt)
            else:
                if f2sec(i - self.lighttime[stimulate],fps) > 4.5:
                    flag = 0
                    stimulate+=1
                elif abs(i - self.lighttime[stimulate]) < 2:
                    plt.scatter(i,k,c='r')
                    # self.pAngle.circle(i,k,size=10, line_color="navy", fill_color="red", fill_alpha=0.5)
                    self.stimus.append(cnt)
                else:
                    plt.scatter(i,k,c='b')
                    # self.pAngle.circle(i,k,size=10, line_color="white", fill_color="blue", fill_alpha=0.5)
                    self.stimus.append(cnt)
                    
        # print(self.stimus)
        plt.xlabel('number of frame')
        plt.ylabel('angle(deg)')
        plt.title('angle curve')
        self.segment(-20,20)
        plt.savefig('pAngle.png')
        # plt.savefig('squares.png')
        # plt.show()
        # self.pAngle.xaxis.axis_label = "帧序号"
        # self.pAngle.yaxis.axis_label = "角度"
       
        # self.pAngle_interp.xaxis.axis_label = "帧序号"
        # self.pAngle_interp.yaxis.axis_label = "角度"
        
        plt.figure()
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
        
        self.segment(-20,20)
        plt.xlabel('number of frame')
        plt.ylabel('angle(deg)')
        plt.title('interpolate angle curve')
        # output_file(filename="res_Angle.html", title="angle result")
        # save(self.pAngle)
        # save(self.pAngle_interp)
        points = [(self.frames[i],self.Theta[i]) for i in range(len(self.frames)) if i in self.stimus]
        tList = np.linspace(0,1,200)
        # show_curve(points,tList)
        # show_der(points,tList)
        # show(self.pAngle)
        # show(self.pAngle_interp)
        plt.savefig('pAngle_interp.png')
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
        
        plt.figure()
        # self.pOmega = figure(width=600, height=600)
        # plt.scatter(self.interp_x,self.interp_omega,c='black')
        # self.pOmega.circle(self.interp_x,self.interp_omega,size=5,fill_color='black')
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
        plt.savefig('pOmega.png')
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
        # self.pPath = figure(width=600, height=600)
        plt.figure()
        print('frames',self.frames)
        print(len(self.frames))
        flag = 1 if len(self.X1)>0 else 0
        for i,frame in enumerate(self.frames):
            if self.minDis(frame) < 1:
                print(frame)
                print(self.X_mid[i],self.Y_mid[i])
                plt.scatter(self.X_mid[i][1], self.Y_mid[i][1],c='r')
                if flag:
                    plt.scatter(self.X1[i], self.Y1[i], c='r')
                    plt.scatter(self.X2[i], self.Y2[i], c='r')
                # self.pPath.circle(self.X_mid[i][1], self.Y_mid[i][1], fill_color="red", size=8)
                # self.pPath.circle(self.X1[i], self.Y1[i], fill_color="red", size=8)
                # self.pPath.circle(self.X2[i], self.Y2[i], fill_color="red", size=8)
                
        plot_xmid = [i[1] for i in self.X_mid if self.in_range(i[0])]
        plot_ymid = [i[1] for i in self.Y_mid if self.in_range(i[0])]
        plt.plot(plot_xmid, plot_ymid,c='b')
        # self.pPath.line(plot_xmid, plot_ymid, line_color="blue", line_alpha=0.6, line_width=2)
        if flag:
            plot_xf = [self.X1[i] for i in range(self.num) if self.frames[i] in self.stimus]
            plot_yf = [self.Y1[i] for i in range(self.num) if self.frames[i] in self.stimus]
            plot_xb = [self.X2[i] for i in range(self.num) if self.frames[i] in self.stimus]
            plot_yb = [self.Y2[i] for i in range(self.num) if self.frames[i] in self.stimus]
        
            plt.plot(plot_xf,plot_yf,c='green')
            # self.pPath.line(plot_xf,plot_yf, line_color="green", line_alpha=0.6, line_width=2)
            plt.plot(plot_xb,plot_yb,c='purple')
            # self.pPath.line(plot_xb,plot_yb, line_color="purple", line_alpha=0.6, line_width=2)
        # output_file(filename="res_Path.html", title="path result")
        # save(self.pPath)
        # show(self.pPath)
        plt.title('path curve')
        plt.savefig('pPath.png')
        plt.show()
        
    def showCurve(self,):
        # self.pCurve = figure(width=600, height=600)
        plt.figure()
        max_r = 0
        for i in range(len(self.X_mid) - 2):
            if self.X_mid[i+2][0]-self.X_mid[i+1][0] == 1 and self.X_mid[i+1][0]-self.X_mid[i][0] == 1: # 连续三点
                d_s = sqrt((self.Y_mid[i+1][1]-self.Y_mid[i][1])**2 +(self.X_mid[i+1][1]-self.X_mid[i][1])**2)
                if d_s > 0.01:
                    alpha1 = (atan((self.Y_mid[i+2][1] - self.Y_mid[i+1][1]) / (self.X_mid[i+2][1] - self.X_mid[i+1][1])) 
                            if abs(self.X_mid[i+2][1] - self.X_mid[i+1][1]) > 0.1 else pi/2)
                    alpha2 = (atan((self.Y_mid[i+1][1] - self.Y_mid[i][1]) / (self.X_mid[i+1][1] - self.X_mid[i][1])) 
                            if abs(self.X_mid[i+1][1] - self.X_mid[i][1]) > 0.1 else pi/2)
                    d_alpha = alpha1 - alpha2
                    if d_alpha > 0.001:
                        r = d_s / d_alpha
                        if self.minDis(self.X_mid[i][0]) < 3:
                            plt.scatter(self.X_mid[i][0], r, c='r')
                            # self.pCurve.circle(self.X_mid[i][0], r, line_color="white", fill_color="red", fill_alpha=1, size=10)
                        else:
                            plt.scatter(self.X_mid[i][0], r, c='b')
                            # self.pCurve.circle(self.X_mid[i][0], r, line_color="white", fill_color="blue", fill_alpha=1, size=10)
                        if r > max_r:
                            max_r = r
                    else:
                        pass
                else:
                    pass
        self.segment(0,max_r)
        # output_file(filename="res_Curvature.html", title="radius of curvature result")
        # save(self.pCurve)
        # show(self.pCurve)
        plt.title('turning radius')
        plt.savefig('pCurve.png')
        plt.show()
        
    def showDist(self):
        # self.pDist = figure(width=400, height=400)
        for each in self.D:
            self.pDist.circle(each[0], each[1], line_color="white", fill_color="blue", fill_alpha=0.5)
        self.pDist.xaxis.axis_label = "帧序号"
        self.pDist.yaxis.axis_label = "距离（像素）"
        # show(self.pDist)
        
if __name__ == '__main__':
    data_dealer = Dealer()
    data_dealer.deal_time(open('out-light-every.txt','r'), 30)
    # data_dealer.parse_fbpoints(open('out-meanshift-1.txt','r'),open('out-meanshift-2.txt','r'),30)
    data_dealer.parse_center_angle(open('out-contour-center.txt','r'),open('out-contour-theta.txt','r'),30)
    # data_dealer.showPath()
    # data_dealer.showCurve()
    data_dealer.showAngle(30)
    # data_dealer.showOmega(30)
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
            
    