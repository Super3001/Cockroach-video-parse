import cv2
import tkinter as tk
import numpy as np
from utils import dcut
from tkinter.messagebox import askyesno, askyesnocancel

"""debug global property"""
from control import pstatus
# pstatus = "release"
# pstatus = "debug"

use_global = True
"""@deprecated global vars"""
if use_global:
    g_rect = (0,)*4
    minis = []
    end_process = False

""" 递归进行选框，功能未完成 """
'''min_x,min_y,width,height = g_rect # (x,y,w,h)
frame = frame[min_y:min_y + height, min_x:min_x + width]
frame = cv2.resize(frame,(1200,800))
cv2.destroyWindow('roi')
minis.append(g_rect)
return monitor_show(frame,ratio,center_point,time,function,container)'''

class Tractor: 
    def __init__(self) -> None:
        """data"""
        self.gbColor = None
        self.gbRect = None
        self.gbPoint = None
        self.gbInput = None
        """property for calculate"""
        self.mutiple = 1
        """property for show"""
        self.cut_edge = 50
        self.status = 'none' # ['none', 'cancel', 'done']
        
    def monitor_show(self, frame, ratio=0, center_point=(-1,-1), time=0, function=None, container=None, reset_function=None):
        # print(center_point)
        frame_show = frame.copy()
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
            # cv2.imshow("window",frame_cut)
            cv2.imshow("image",cv2.resize(frame_cut,(1200,800)))
            
        elif ratio > 0 and ratio <= 1:
            cv2.imshow("image",cv2.resize(frame,(1200,800)))
            
        else:
            cv2.imshow("image",frame)
        if function:
            cv2.setMouseCallback("image", function, frame_show)
        key = cv2.waitKey(time)
        print(key)
        if key == ord('q'):
            return 1
        elif key == 13: # ENTER键
            cv2.destroyAllWindows()
            return 0
        else:
            '''取消：重新选择'''
            if reset_function is not None:
                reset_function()
                self.monitor_show(frame, ratio, center_point, time, function, container, reset_function)
            return -1
        
    def set(self, ord, value):
        if ord == 'mutiple':
            self.mutiple = value 

    def drawCircle(self,event,x,y,flags,frame):
        if event == cv2.EVENT_LBUTTONDOWN:
            # cv2.circle(frame, (x,y), 5, (255, 0, 0), -1)
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
                cv2.destroyAllWindows()
            else:
                cv2.destroyWindow("Point")
            
    """@former: pointColor
    def pointColor(self,event,x,y,flags,frame):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x,y)
            # global gbColor
            self.gbColor = frame[y][x]
    """
    
    def pointColor(self,event,x,y,flags,frame):
        end_process = False
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
                    self.status = "done"
                else:
                    self.status = "waiting"
            elif key == ord('q'):
                cv2.destroyAllWindows()
            else:
                if cv2.getWindowProperty("color", cv2.WND_PROP_VISIBLE) >= 0:
                    cv2.destroyWindow("color")
    
    def mouse_rect(self,event,x,y,flags,frame):
        frame_show = frame.copy()
        global point1, point2, g_rect, cut_img
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击,则在原图打点
            # print("1-EVENT_LBUTTONDOWN")
            point1 = (x, y)
            cv2.circle(frame_show, point1, 10, (0, 255, 0), 5)
            cv2.imshow("image", frame_show)
    
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳，画框
            # print("2-EVENT_FLAG_LBUTTON")
            cv2.rectangle(frame_show, point1, (x, y), (255, 0, 0), thickness=2)
            cv2.imshow("image", frame_show)
    
        elif event == cv2.EVENT_LBUTTONUP:  # 左键释放，显示
            # print("3-EVENT_LBUTTONUP")
            point2 = (x, y)
            cv2.rectangle(frame_show, point1, point2, (0, 0, 255), thickness=2)
            cv2.imshow("image", frame_show)
            key = cv2.waitKey(0) & 0xFF
            if key == 13:
                cv2.destroyAllWindows()
            else:
                cv2.destroyWindow("image")
            if point1!=point2:
                min_x = min(point1[0], point2[0])
                min_y = min(point1[1], point2[1])
                width = abs(point1[0] - point2[0])
                height = abs(point1[1] - point2[1])
                g_rect=[min_x,min_y,width,height] # (x,y,w,h)
                cut_img = frame[min_y:min_y + height, min_x:min_x + width]
                # cv2.imshow('roi',cut_img)
                # res = askyesno('COnfirm',f'Rect: {point1} : {point2} \nconfirm?')
                # if res == 1:
                #     cv2.destroyAllWindows()
            else:
                cv2.destroyWindow("image")
        else:
            pass
                
        """debug"""
        if event == 10: # cv2.EVENT_MOUSEWHEEL = 10
            print('pos:', (x, y))
            print('color:', img[y][x])
            print('flags:', flags)
            print(f'gbRect: {self.gbRect}')
    
    """@former: mouse_rect
    def firstRect(self,event,x,y,flags,frame):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x,y)
            # global gbRect
            if len(self.gbRect)<=1:
                self.gbRect.append((x,y))
            else:
                temp = self.gbRect[1]
                self.gbRect = []
                self.gbRect.append(temp)
                self.gbRect.append((x,y))  
    """
    
    """changed"""
    def tract_color(self, frame):
        self.gbColor = [0,0,0]
        # cv2.imshow("windowName",frame)
        # cv2.setMouseCallback("windowName", self.pointColor, frame)
        # cv2.waitKey(0)
        if self.monitor_show(frame, function=self.pointColor, reset_function=self.color_reset):
            cv2.destroyAllWindows()
            return (-1,-1,-1)
        print(self.gbColor)
        return self.gbColor
    
    def color_reset(self):
        cv2.destroyWindow("roi")

    """changed"""
    def select_rect(self, frame):
        # global gbRect
        self.gbRect = []
        cv2.imshow("windowName",frame)
        cv2.setMouseCallback("windowName", self.mouse_rect, frame)
        key = cv2.waitKey(0)
        
        if key == ord('q'):
            cv2.destroyAllWindows()
            return (None,)*4
        elif key == 13:
            cv2.destroyAllWindows()
        
        # print(self.mutiple)
        # A,B = self.gbRect
        """alter order of the pos and deal mutiply"""
        global point1, point2
        x1,y1 = point1
        x2,y2 = point2
        x_start = min(x1, x2)
        x_end = max(x1, x2)
        y_start = min(y1, y2)
        y_end = max(y1, y2)
        self.gbRect = ((int(x_start*self.mutiple), int(y_start*self.mutiple)), (int(x_end*self.mutiple), int(y_end*self.mutiple)))
        print(self.gbRect)
        return self.gbRect[0][1],self.gbRect[1][1]-self.gbRect[0][1],self.gbRect[0][0],self.gbRect[1][0]-self.gbRect[0][0] # y, x, h, w

    def tractPoint(self,frame):
        self.gbPoint = (-1,-1)
        cv2.imshow("windowName",frame)
        cv2.setMouseCallback("windowName", self.pointPos, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('mutiple', self.mutiple)
        x,y = self.gbPoint
        self.gbPoint = (x*self.mutiple, y*self.mutiple)
        return
                
    """ @deprecated: using QtUI """
    '''def inputbox(self,show_text):
        app = QApplication(sys.argv)
        ex = Inputbox()
        ex.getText(show_text)
        self.gbInput = ex.text
        app.exit(0)'''

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

'''class Inputbox(QWidget):
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
                self.close()'''
                
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
    def __init__(self) -> None:
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
        global cut_img,K
        h,w,c = cut_img.shape
        gray = cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)
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
            K = np.sin( (gray - mean) / std * np.pi / 2)
            
            # normalize
            K = self.normalize(K)
    
    def select_window(self,frame):
        global minis
        _rtn = -1
        while _rtn < 0:
            _rtn = self.monitor_show(frame, function = self.mouse, container=self)
            if _rtn == 1:
                cv2.destroyAllWindows()
                return 'q'
            elif _rtn == 0:
                self.generate_kernal(kind='sin')
                self.K = K
                return g_rect, minis
            else:
                cv2.destroyWindow("roi")            
        
    def mouse(self,event,x,y,flags,frame):
        frame_show = frame.copy()
        global point1, point2, g_rect, cut_img
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击,则在原图打点
            # print("1-EVENT_LBUTTONDOWN")
            point1 = (x, y)
            cv2.circle(frame_show, point1, 10, (0, 255, 0), 5)
            cv2.imshow("image", frame_show)
    
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳，画框
            # print("2-EVENT_FLAG_LBUTTON")
            cv2.rectangle(frame_show, point1, (x, y), (255, 0, 0), thickness=2)
            cv2.imshow("image", frame_show)
    
        elif event == cv2.EVENT_LBUTTONUP:  # 左键释放，显示
            # print("3-EVENT_LBUTTONUP")
            point2 = (x, y)
            cv2.rectangle(frame_show, point1, point2, (0, 0, 255), thickness=2)
            cv2.imshow("image", frame_show)
            if point1!=point2:
                min_x = min(point1[0], point2[0])
                min_y = min(point1[1], point2[1])
                width = abs(point1[0] - point2[0])
                height = abs(point1[1] - point2[1])
                g_rect=[min_x,min_y,width,height] # (x,y,w,h)
                cut_img = frame[min_y:min_y + height, min_x:min_x + width]
                h_, w_, _ = cut_img.shape
                out_h = 600
                cv2.imshow('roi', cv2.resize(cut_img, (int(out_h/h_ * w_), out_h)))
                """ cv.resize的第二个参数的格式是(w, h) """
                
            else:
                # cv2.destroyWindow("image")
                pass
        else:
            pass

if pstatus == "debug":
    cap = cv2.VideoCapture(r"D:\Github\Cockroach-video-parse\src\DSC_2059.mp4")
    ret, img = cap.read()
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if __name__ == '__main__':
        # Idf = Identifier()
        # Idf.select_window(img)
        Trc = Tractor()
        Trc.tract_color(img)
        