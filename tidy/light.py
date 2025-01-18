# light.py
import cv2 as cv
from tract_point import Tractor
from processing import my_show
from control import pstatus
from utils import Stdout_progressbar
       
""" 提取闪光主函数，不跳读 """
def tractLight(cap, master, OutWindow, progressBar, thres=150, show_time=1000//30):
    """
    Args:
        cap: cv.VideoCapture
        master: tkinter.Tk
        OutWindow: OutWindow
        progressBar: tkinter.ttk.Progressbar
        thres: int - (0,255)
        show_time: int - 毫秒
    """
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    num = cap.get(7)
    ret, frame0 = cap.read()
    
    Trc = Tractor()
    _rtn = Trc.tractPoint(cv.resize(frame0,(1200,800)),"click the light position, press enter to confirm, q to quit, space to redo")
    x,y = Trc.gbPoint

    if _rtn == 'quit' or x == -1 or y == -1:
        return 'quit'
    
    domain = [y-3,y+3,x-3,x+3] # 上，下，左，右
    file = open('out-light-every.txt','w')
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    if OutWindow and OutWindow.display:
        OutWindow.textboxprocess.delete('0.0','end')
        OutWindow.textboxprocess.insert('0.0',"闪光帧序号：\n")
        OutWindow.master.lift()
    cnt = 0
    progressBar['maximum'] = num
    stdoutpb = Stdout_progressbar(num, not(OutWindow and OutWindow.display))
    stdoutpb.reset()
    while True:
        success, frame = cap.read()
        if success:
            frame = cv.resize(frame,(1200,800))
            cnt += 1
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
                    frame_show = frame.copy()
                    cv.rectangle(frame_show, (domain[2]-20,domain[0]-20),(domain[3]+20,domain[1]+20),(0,0,255),2)
                    if my_show(frame_show, _time=show_time):
                        return 'stop'
                    """使用帧数进行记录"""
                else:
                    file.write(f'{cnt}\n')
            else:
                if OutWindow and OutWindow.display:
                    frame_show = frame.copy()
                    cv.rectangle(frame_show, (domain[2]-20,domain[0]-20),(domain[3]+20,domain[1]+20),(255,0,0),2)
                    if my_show(frame_show, _time=show_time):
                        return 'stop'
                else:
                    pass
            stdoutpb.update(cnt)

        else:
            stdoutpb.update(-1)
            break
    return 'OK'
