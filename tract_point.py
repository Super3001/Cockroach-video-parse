import cv2 as cv
from tkinter.messagebox import *
import tkinter as tk
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit
from PyQt5.QtGui import QIcon
import numpy as np

g_rect = (0,)*4
cut_img = None
minis = []

def monitor_show(frame, ratio=0, center_point=(-1,-1), time=0, function=None, container=None):
    # print(center_point)
    height = frame.shape[0]
    width = frame.shape[1]
    global g_rect
    # print(height,width)
    # obj = frame[int(height*percentage[0]):int(height*percentage[1]),int(width*percentage[2]):int(width*percentage[3])]
    if ratio > 1:
        up,down,left,right = int(center_point[1] - 400/ratio),int(center_point[1] + 400/ratio), int(center_point[0] - 600/ratio),int(center_point[0] + 600/ratio)
        # print(up,down,left,right)
        frame_cut = frame[up:down,left:right]
        # print('magnified')
        # cv.imshow("window",frame_cut)
        cv.imshow("image",cv.resize(frame_cut,(1200,800)))
        
    elif ratio > 0 and ratio <= 1:
        cv.imshow("image",cv.resize(frame,(1200,800)))
        
    else:
        cv.imshow("image",frame)
    if function:
        cv.setMouseCallback("image",function,frame)
    key = cv.waitKey(time)
    if key == ord('q'):
        return 1
    elif key == 13: # ENTER键
        container.square = g_rect
        return 0
    else:
        min_x,min_y,width,height = g_rect # (x,y,w,h)
        frame = frame[min_y:min_y + height, min_x:min_x + width]
        frame = cv.resize(frame,(1200,800))
        cv.destroyWindow('roi')
        minis.append(g_rect)
        return monitor_show(frame,ratio,center_point,time,function,container)

class Tractor:
    
    def __init__(self) -> None:
        self.gbColor = None
        self.gbRect = None
        self.gbPoint = None
        self.gbInput = None
        self.mutiple = 1
        
    def set(self, ord, value):
        if ord == 'mutiple':
            self.mutiple = value 

    def drawCircle(self,event,x,y,flags,frame):
        if event == cv.EVENT_LBUTTONDOWN:
            # cv.circle(frame, (x,y), 5, (255, 0, 0), -1)
            print(x,y)
            print(frame[y][x])
            
    def pointPos(self,event,x,y,flags,frame):
        if event == cv.EVENT_LBUTTONDOWN:
            print(x,y)
            self.gbPoint = (x,y)
            
    def pointColor(self,event,x,y,flags,frame):
        if event == cv.EVENT_LBUTTONDOWN:
            print(x,y)
            # global gbColor
            self.gbColor = frame[y][x]
            
    def firstRect(self,event,x,y,flags,frame):
        if event == cv.EVENT_LBUTTONDOWN:
            print(x,y)
            # global gbRect
            if len(self.gbRect)<=1:
                self.gbRect.append((x,y))
            else:
                temp = self.gbRect[1]
                self.gbRect = []
                self.gbRect.append(temp)
                self.gbRect.append((x,y))  
    
    def tract_color(self,frame):
        cv.imshow("windowName",frame)
        cv.setMouseCallback("windowName", self.pointColor, frame)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return self.gbColor

    def select_rect(self,frame):
        # global gbRect
        self.gbRect = []
        cv.imshow("windowName",frame)
        cv.setMouseCallback("windowName", self.firstRect, frame)
        key = cv.waitKey(0)
        cv.destroyAllWindows()
        if key == ord('q'):
            return (None,)*4
        if len(self.gbRect) < 2:
            showinfo('请提取两个点')
            return self.select_rect(frame)
        
        print(self.mutiple)
        A,B = self.gbRect
        x1,y1 = A
        x2,y2 = B
        self.gbRect = ((int(x1*self.mutiple), int(y1*self.mutiple)), (int(x2*self.mutiple), int(y2*self.mutiple)))
        print(self.gbRect)
        return self.gbRect[0][1],self.gbRect[1][1]-self.gbRect[0][1],self.gbRect[0][0],self.gbRect[1][0]-self.gbRect[0][0]

    def tractPoint(self,frame):
        cv.imshow("windowName",frame)
        cv.setMouseCallback("windowName", self.pointPos, frame)
        cv.waitKey(0)
        cv.destroyAllWindows()
        print(self.mutiple)
        x,y = self.gbPoint
        self.gbPoint = (x*self.mutiple, y*self.mutiple)
        return
                
    def inputbox(self,show_text):
        app = QApplication(sys.argv)
        ex = Inputbox()
        ex.getText(show_text)
        self.gbInput = ex.text
        app.exit(0)

class Inputbox(QWidget):
        def __init__(self):
            super().__init__()
            self.title = '输入框'
            self.left = 600
            self.top = 300
            self.width = 640
            self.height = 480
            self.text = ''
            self.initUI()
            
        def initUI(self):
            self.setWindowTitle(self.title)
            self.setGeometry(self.left, self.top, self.width, self.height)

        def getText(self, show_text):
            self.text, okPressed = QInputDialog.getText(self, "Get text",show_text, QLineEdit.Normal, "")
            if okPressed and self.text != '':
                print(self.text)
                self.close()
                
def conv2d(bitmap, kernal):
    if bitmap.shape[0] != kernal.shape[0]:
        raise ValueError('Non-equal')
    n = kernal.shape[0]
    res = 0
    for i in range(n):
        for j in range(n):
            res += bitmap[i][j]*kernal[i][j]
    return res
                
class Identifier:
    def __init__(self,frame) -> None:
        self.frame = frame
        self.select_window(frame)
        self.Types = ['cross','add','square']
    
    def parse(self,bitmap):
        n = bitmap.shape[0]
        max_value = 0
        max_type = None
        max_K = None
        for each in self.Types:
            K = self.generate_kernal(n,each)
            value = conv2d(bitmap,K)
            if value > max_value:
                max_value = value
                max_type = each
                max_K = K
        return max_type,max_K
    
    def evaluate(self,x):
        return -np.cos(x*np.pi/255)
    
    def generate_kernal(self):
        global cut_img,K
        w,h,c = cut_img.shape()
        if w != h:
            raise ValueError("not a square")
        K = np.zeros((h,w))
        for i in range(h):
            for j in range(w):
                K[i] [j] = self.evaluate(cut_img[i][j])
    
    def select_window(self,frame):
        minis = []
        if monitor_show(frame, function = self.mouse, container=self):
            return 'q'
        x,y,w,h = self.square
        self.generate_kernal()
        self.K = K
        return g_rect,minis
        
    def mouse(self,event,x,y,flags,frame):
        frame_show = frame.copy()
        global point1, point2, g_rect, cut_img
        if event == cv.EVENT_LBUTTONDOWN:  # 左键点击,则在原图打点
            print("1-EVENT_LBUTTONDOWN")
            point1 = (x, y)
            cv.circle(frame_show, point1, 10, (0, 255, 0), 5)
            cv.imshow("image", frame_show)
    
        elif event == cv.EVENT_MOUSEMOVE and (flags & cv.EVENT_FLAG_LBUTTON):  # 按住左键拖曳，画框
            print("2-EVENT_FLAG_LBUTTON")
            cv.rectangle(frame_show, point1, (x, y), (255, 0, 0), thickness=2)
            cv.imshow("image", frame_show)
    
        elif event == cv.EVENT_LBUTTONUP:  # 左键释放，显示
            print("3-EVENT_LBUTTONUP")
            point2 = (x, y)
            cv.rectangle(frame_show, point1, point2, (0, 0, 255), thickness=2)
            cv.imshow("image", frame_show)
            if point1!=point2:
                min_x = min(point1[0], point2[0])
                min_y = min(point1[1], point2[1])
                width = abs(point1[0] - point2[0])
                height = abs(point1[1] - point2[1])
                g_rect=[min_x,min_y,width,height] # (x,y,w,h)
                cut_img = frame[min_y:min_y + height, min_x:min_x + width]
                cv.imshow('roi',cut_img)
        else:
            pass
        

cap = cv.VideoCapture("C:\\Users\\LENOVO\\Desktop\\10Hz，左，样本3 00_00_00-00_00_19.40_Trim.mp4")
# 获取第一帧位置，并指定目标位置
ret, img = cap.read()
size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), 
        int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
print(size)

if __name__ == '__main__':
    Idf = Identifier(img)
    # Idf.select_window(Idf.frame)