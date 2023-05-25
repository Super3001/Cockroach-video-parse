import cv2 as cv
from tract_point import *

# pstatus = "release"
pstatus = "debug"

def legal(x,y,width,height):
    if x < 0 or x >= width:
        return 0
    if y < 0 or y >= height:
        return 0
    return 1
       
def tractLight(cap, master, OutWindow, progressBar, thres=150):
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    # w, h = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), 
            # int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    num = cap.get(7)
    # print(cap)
    ret, frame0 = cap.read()
    # print(frame0)
    Trc = Tractor()
    Trc.tractPoint(cv.resize(frame0,(1200,800)))
    x,y = Trc.gbPoint
    domain = [y-3,y+3,x-3,x+3] # 上，下，左，右
    file = open('out-light-every.txt','w')
    # cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    if OutWindow and OutWindow.display:
        OutWindow.textboxprocess.delete('0.0','end')
        OutWindow.textboxprocess.insert('0.0',"闪光帧序号：\n")
    cnt = 0
    progressBar['maximum'] = num
    while True:
        success, frame = cap.read()
        if success:
            frame = cv.resize(frame,(1200,800))
            progressBar['value'] = cnt
            master.update()
            max_value = 0
            gray = cv.cvtColor(frame[domain[0]:domain[1]+1 , domain[2]:domain[3]+1],cv.COLOR_BGR2GRAY)
            for i in range(7):
                for j in range(7):
                    max_value = max(max_value,gray[i][j])
                        
            if max_value > thres:
                if OutWindow and OutWindow.display:
                    OutWindow.textboxprocess.insert("0.0","%d\n" % cnt)
                """使用帧数进行记录"""
                file.write(f'{cnt}\n')
                # file.write('1\n')
            else:
                # file.write('0\n')
                pass
            cnt += 1
            
        else:
            break
        
if pstatus == "debug":
    class FakeMs:
        def __init__(self) -> None:
            self.cnt = 0
        
        def update(self):
            self.cnt += 1

    if __name__ == '__main__':
        cap = cv.VideoCapture('C:\\Users\\LENOVO\\Desktop\\10Hz,右，样本2 00_00_51-00_01_46.mp4')
        tractLight(cap, master=FakeMs(),OutWindow=None,progressBar=dict())
