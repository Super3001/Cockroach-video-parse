# tract_point.py

import cv2
import tkinter as tk
import numpy as np
from utils import dcut
from tkinter.messagebox import askyesno
from signals import *

"""debug global property"""
from control import pstatus

use_global = False # no longer use global vars

class Tractor:
    def __init__(self) -> None:
        '''init a Tractor'''

        """data"""
        self.gbColor = None
        self.gbRect = None
        self.gbPoint = None
        self.gbInput = None
        """property for calculate"""
        self.mutiple = 1
        """property for show"""
        self.cut_edge = 50
        """working status"""
        self.status = 'none' # ['none', 'cancel', 'done', 'waiting']
        
    def monitor_show(self, frame, ratio=1, time=0, function=None, reset_function=None):
        height = frame.shape[0]
        width = frame.shape[1]
        frame = cv2.resize(frame, (int(width*ratio), int(height*ratio)))
        
        cv2.imshow("image", frame)
        if function is not None:
            cv2.setMouseCallback("image", function, frame)
        key = cv2.waitKey(time)
        # print(key)
        if key == -1:
            '''代表已经处理过了'''
            # print(self.status)
            return -1 # 代表unknown(?)
        if key == ord('q'):
            '''取消：退出选择'''
            self.status = 'cancel'
            return 1
        elif key == 13:
            self.status = 'done'
            return 0
        else:
            self.status = 'waiting'
            '''取消：重新选择'''
            if reset_function is not None:
                reset_function()
                _rtn = self.monitor_show(frame, ratio, time, function, reset_function)
            return _rtn
        
    def set(self, ord, value):
        if ord == 'mutiple':
            self.mutiple = value
            
    def txt(self, image, text):
        # 设置文字参数
        # text = "Hello, OpenCV!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color = (255, 255, 255)  # 文字颜色，BGR格式
        thickness = 2  # 文字线条粗细

        # 在图像上添加文字
        image_height, image_width = image.shape[:2]
        text_width, text_height = cv2.getTextSize(text, font, font_scale, thickness)[0]

        # 计算文本在图像底部10%位置居中时的坐标
        text_x = int((image_width - text_width) / 2)
        text_y = int(image_height * 0.9) + int((image_height * 0.1 - text_height) / 2)

        cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)
            
    def drawCircle(self,event,x,y,flags,frame):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(frame, (x,y), 5, (255, 0, 0), -1)
            
            print(x,y)
            print(frame[y][x])
            
    def pointPos(self,event,x,y,flags,frame):
        frame_show = frame.copy()
        edge = self.cut_edge
        if event == cv2.EVENT_LBUTTONDOWN:
            self.gbPoint = (x,y)
            cv2.circle(frame_show,self.gbPoint,1,(0,0,255),1)
            x1,x2 = max(x-edge,0), min(x+edge,frame.shape[1]-1)
            y1,y2 = max(y-edge,0), min(y+edge,frame.shape[0]-1)
            cv2.imshow("Point",cv2.resize(frame_show[y1:y2,x1:x2], (800,800)))
            cv2.moveWindow("Point",x-edge//2,y-edge//2)
            key = cv2.waitKey(0) & 0xFF
            if key == 13: # ENTER
                print(self.gbPoint)
                self.status = 'done'
                cv2.destroyAllWindows()
            else:
                self.status = 'waiting'
                cv2.destroyWindow("Point")
            
    def pointColor(self,event,x,y,flags,frame):
        frame_show = frame.copy()
        if event == cv2.EVENT_LBUTTONDOWN:
            """debug"""
            # print(x, y)
            self.gbColor = frame[y][x]
            cv2.circle(frame_show,(x,y),1,(0,0,255),1)
            frame_show = dcut(frame_show, (y-30,y+30,x-30,x+30))
            cv2.imshow("color",cv2.resize(frame_show, (400,400)))
            key = cv2.waitKey(0) & 0xFF
            if key == 13: # ENTER
                res = askyesno('Confirm',f'Color(BGR): {self.gbColor} \nconfirm?')
                if res == 1:
                    # print("done")
                    self.status = "done"
                    cv2.destroyAllWindows()
                else:
                    self.status = "waiting"
                    cv2.destroyWindow("color")
            elif key == ord('q'):
                self.status = 'cancel'
                cv2.destroyAllWindows()
            # elif key == -1:
            elif key == 255:
                print('previous unhandle:')
                print('status:', self.status)
            else:
                # print(key)
                self.status = 'waiting'
                cv2.destroyWindow("color")
    
    def mouse_rect(self,event,x,y,flags,frame):
        frame_show = frame.copy()
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击,则在原图打点
            # print("1-EVENT_LBUTTONDOWN")
            self.point1 = (x, y)
            cv2.circle(frame_show, self.point1, 10, (0, 255, 0), 5)
            cv2.imshow("image", frame_show)
    
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳，画框
            # print("2-EVENT_FLAG_LBUTTON")
            cv2.rectangle(frame_show, self.point1, (x, y), (255, 0, 0), thickness=2)
            cv2.imshow("image", frame_show)
    
        elif event == cv2.EVENT_LBUTTONUP:  # 左键释放，显示
            # print("3-EVENT_LBUTTONUP")
            self.point2 = (x, y)
            cv2.rectangle(frame_show, self.point1, self.point2, (0, 0, 255), thickness=2)
            cv2.imshow("image", frame_show)
            key = cv2.waitKey(0) & 0xFF
            if key == 13:
                cv2.destroyAllWindows()
            else:
                cv2.destroyWindow("image")
            if self.point1!=self.point2:
                min_x = min(self.point1[0], self.point2[0])
                min_y = min(self.point1[1], self.point2[1])
                width = abs(self.point1[0] - self.point2[0])
                height = abs(self.point1[1] - self.point2[1])
                self.gbRect=[min_x,min_y,width,height] # (x,y,w,h)
                self.cut_img = frame[min_y:min_y + height, min_x:min_x + width]
                # cv2.imshow('roi',cut_img)
                # res = askyesno('COnfirm',f'Rect: {point1} : {self.point2} \nconfirm?')
                # if res == 1:
                #     cv2.destroyAllWindows()
            else:
                cv2.destroyWindow("image")
        else:
            pass
                
        """debug"""
        if event == 10: # cv2.EVENT_MOUSEWHEEL = 10
            print('pos:', (x, y))
            print('color:', frame[y][x])
            print('flags:', flags)
            print(f'gbRect: {self.gbRect}')
           
    """color""" 
    def tract_color(self, frame):
        self.gbColor = [0,0,0]
        # cv2.imshow("windowName",frame)
        # cv2.setMouseCallback("windowName", self.pointColor, frame)
        # cv2.waitKey(0)
        self.txt(frame, color_prompt)
        if self.monitor_show(frame, function=self.pointColor, reset_function=self.color_reset) == 1:
            cv2.destroyAllWindows()
            return (-1,-1,-1)
        print(self.gbColor)
        return self.gbColor
    
    def color_reset(self):
        cv2.namedWindow("roi") # 防止报错
        cv2.destroyWindow("roi")
        
    """meanshift contour"""
    def select_rect(self, frame):
        self.gbRect = []
        self.txt(frame, select_window_prompt)
        cv2.imshow("windowName",frame)
        cv2.setMouseCallback("windowName", self.mouse_rect, frame)
        key = cv2.waitKey(0)
        
        if key == ord('q'):
            cv2.destroyAllWindows()
            return (None,)*4
        elif key == 13:
            cv2.destroyAllWindows()
        
        """alter order of the pos and deal mutiply"""
        x1,y1 = self.point1
        x2,y2 = self.point2
        x_start = min(x1, x2)
        x_end = max(x1, x2)
        y_start = min(y1, y2)
        y_end = max(y1, y2)
        self.gbRect = ((int(x_start*self.mutiple), int(y_start*self.mutiple)), (int(x_end*self.mutiple), int(y_end*self.mutiple)))
        print(self.gbRect)
        return self.gbRect[0][1],self.gbRect[1][1]-self.gbRect[0][1],self.gbRect[0][0],self.gbRect[1][0]-self.gbRect[0][0] # y, x, h, w
    
    """light magnify"""
    def tractPoint(self,frame, _text):
        self.gbPoint = (-1,-1)
        self.txt(frame, _text)
        cv2.imshow("windowName",frame)
        cv2.setMouseCallback("windowName", self.pointPos, frame)
        key = cv2.waitKey(0)
        if key == ord('q'):
            return 'quit'
        cv2.destroyAllWindows()
        # print('mutiple', self.mutiple)
        x,y = self.gbPoint
        self.gbPoint = (x*self.mutiple, y*self.mutiple)
        return 'ok'
    
    def inputbox(self, root, show_text):
        self.gbInput = None # 防止得到错误的输入
        input_window = tk.Toplevel(root)
        self.input_window = input_window
        
        label = tk.Label(input_window, text=show_text)
        label.pack()
        
        entry = tk.Entry(input_window)
        entry.pack()
        
        # 创建按钮
        button = tk.Button(input_window, text="提交", command=lambda: self.get_user_input(entry.get()))
        button.pack()
        input_window.mainloop()
        return 1
    
    def get_user_input(self, s):
        self.gbInput = s
        self.input_window.quit()
        
class Identifier(Tractor):
    def __init__(self) -> None:
        super().__init__()

    def conv2d(bitmap, kernal):
        if bitmap.shape[0] != kernal.shape[0]:
            raise ValueError('Non-equal')
        n = kernal.shape[0]
        res = 0
        for i in range(n):
            for j in range(n):
                res += bitmap[i][j]*kernal[i][j]
        return res
    
    def parse(self,bitmap):
        n = bitmap.shape[0]
        max_value = 0
        max_type = None
        max_K = None
        for each in self.Types:
            K = self.generate_kernal(n,each)
            value = self.conv2d(bitmap,K)
            if value > max_value:
                max_value = value
                max_type = each
                max_K = K
        return max_type,max_K
    
    def evaluate(self, x, thresh=120):
        # return -np.cos(x*np.pi/255) / 100
        pass

    def normalize(self, mat, significance=100):
        h,w = mat.shape
        pos_sum = 0
        neg_sum = 0  # 绝对值
        for i in range(h):
            for j in range(w):
                if mat[i][j] > 0:
                    pos_sum += mat[i][j]
                else:
                    neg_sum -= mat[i][j]
        if pos_sum == 0 or neg_sum == 0:
            raise ValueError('mat is not bi-chr')
        norm = np.zeros((h,w))
        for i in range(h):
            for j in range(w):
                if mat[i][j] > 0:
                    norm[i][j] = mat[i][j] * significance / pos_sum
                else:
                    norm[i][j] = mat[i][j] * significance / neg_sum
        return norm

    
    """kind: [uniform, sin]"""
    """根据标记点特征产生卷积核"""
    def generate_kernal(self, thresh=120, kind='uniform'):
        h,w,c = self.cut_img.shape
        gray = cv2.cvtColor(self.cut_img, cv2.COLOR_BGR2GRAY)
        if kind == 'uniform':
            pos_n = 0
            neg_n = 0
            for i in range(h):
                for j in range(w):
                    if gray[i][j] >= thresh:
                        pos_n += 1
                    else:
                        neg_n += 1
            K = np.zeros((h,w))
            high = pos_n / (pos_n + neg_n)
            low = -(neg_n / (pos_n + neg_n))
            for i in range(h):
                for j in range(w):
                    if gray[i][j] >= thresh:
                        K[i][j] = high
                    else:
                        K[i][j] = low 
        else:
            # transfrom gray to numpy.array
            gray = np.array(gray)

            # calculate mean and range
            mean = np.mean(gray)
            max_value = np.max(gray)
            min_value = np.min(gray)
            std = max(max_value - mean, mean - min_value)

            # calculate K (altogether) by sin
            self.K = np.sin( (gray - mean) / std * np.pi / 2)
            
            # normalize
            self.K = self.normalize(self.K)
    
    """feature"""
    def select_window(self,frame):
        '''旧的写法'''
        ''' _rtn = -1
            while _rtn < 0:
            _rtn = self.monitor_show(frame, function = self.mouse)
            if _rtn == 1:
                cv2.destroyAllWindows()
                return 'q'
            elif _rtn == 0:
                self.generate_kernal(kind='sin')
                # return g_rect, minis
                return self.gbRect, []
            else:
                cv2.destroyWindow("roi") '''
        
        self.txt(frame, select_window_prompt)
        _rtn = self.monitor_show(frame, function = self.mouse, reset_function = self.idf_reset)
        # if _rtn == 1:
        if self.status == 'cancel':
            print('cancel...')
            cv2.destroyAllWindows()
            return (-1,)*4, []
        # elif _rtn == 0:
        elif self.status == 'done':
            # print('done')
            cv2.destroyAllWindows()
            print(f'{self.gbRect} format:xywh')
            self.generate_kernal(kind='sin')
            # return g_rect, minis
            return self.gbRect, []
        else:
            print("===" * 15)
            print('unknown status... reset to none')
            self.status == 'none'
            return (-2,)*4, []
        
    def idf_reset(self):
        
        # print('reset')
        window_name = "roi"

        # 检查窗口是否存在
        window_status = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)

        if window_status > 0:
            # 关闭窗口
            cv2.destroyWindow(window_name)
        else:
            print('useless key input')

           
    def mouse(self,event,x,y,flags,frame):
        """select_window callback function

        Args:
            event (_type_): _description_
            x (_type_): _description_
            y (_type_): _description_
            flags (_type_): _description_
            frame (_type_): _description_
        """        
        frame_show = frame.copy()
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击,则在原图打点
            # print("1-EVENT_LBUTTONDOWN")
            self.point1 = (x, y)
            cv2.circle(frame_show, self.point1, 10, (0, 255, 0), 5)
            cv2.imshow("image", frame_show)
    
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳，画框
            # print("2-EVENT_FLAG_LBUTTON")
            cv2.rectangle(frame_show, self.point1, (x, y), (255, 0, 0), thickness=2)
            cv2.imshow("image", frame_show)
    
        elif event == cv2.EVENT_LBUTTONUP:  # 左键释放，显示
            # print("3-EVENT_LBUTTONUP")
            self.point2 = (x, y)
            cv2.rectangle(frame_show, self.point1, self.point2, (0, 0, 255), thickness=2)
            cv2.imshow("image", frame_show)
            if self.point1!=self.point2:
                min_x = min(self.point1[0], self.point2[0])
                min_y = min(self.point1[1], self.point2[1])
                width = abs(self.point1[0] - self.point2[0])
                height = abs(self.point1[1] - self.point2[1])
                self.gbRect = [min_x,min_y,width,height] # (x,y,w,h)
                cut_img = frame[min_y:min_y + height, min_x:min_x + width]
                h_, w_, _ = cut_img.shape
                out_h = 600
                cv2.imshow('roi', cv2.resize(cut_img, (int(out_h/h_ * w_), out_h)))
                """ cv.resize的第二个参数的格式是(w, h) """
                self.cut_img = cut_img
            else:
                # cv2.destroyWindow("image")
                pass
        else:
            pass

if pstatus == "debug":
    cap = cv2.VideoCapture(r"D:\Github\Cockroach-video-parse\src\DSC_2059.MOV")
    ret, img = cap.read()
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if __name__ == '__main__':
        Idf = Identifier()
        Idf.select_window(img)
        # Trc = Tractor()
        # Trc.tract_color(img)
