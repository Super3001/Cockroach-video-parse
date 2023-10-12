# utils.py 
import sys
import traceback
import psutil
from pywinauto.application import Application
import time, math
import numpy as np
import matplotlib.pyplot as plt

def timestr():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    # return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

class Stdout_progressbar:
    def __init__(self, max_num, show=True, max_length=50) -> None:
        interval = math.ceil(max_num / max_length)
        length = math.ceil(max_num / interval)
        
        self.interval = interval
        self.length = length
        self.max_num = max_num
        self.time = 0
        self.show = show
        self.skip_n = 1

    def reset(self, skip_n=1):
        self.time = time.perf_counter()
        self.skip_n = skip_n

    def update(self, now_num): # num从1开始
        if not self.show:
            return
        
        if now_num == 1:
            elapse = time.perf_counter() - self.time
            if elapse < 1:
                print(f"{1/elapse:.1f} step(frame) per second, eta {elapse*(self.max_num - 1)/self.skip_n:.1f}s")
            else:
                print(f"{elapse:.1f} second per step(frame), eta {elapse*(self.max_num - 1)/self.skip_n:.1f}s")

        # if now_num % self.interval == 0:
        # 每一次都更新
        elapse = time.perf_counter() - self.time
        i = now_num // self.interval
        percentage = round(now_num / self.max_num * 100)
        print("\r", end="")
        print("Process: {}%: |".format(percentage), "-" * (i), end="")
        print(" "*(self.length - i),"|",f"use {elapse:.1f}s   ", end="")
        sys.stdout.flush()

        if now_num == -1: # 代表结束
            elapse = time.perf_counter() - self.time
            # if now_num % self.interval != 0:
            #     print("\r", end="")
            #     print(f"Progress: {percentage}%: |", "-" * self.length, "|", end="")

            print("\nprocess finished!")
            print(f"totally use {elapse:.1f}s")
            sys.stdout.flush()

def legal(x,y,width,height):
    if x < 0 or x >= width:
        return 0
    if y < 0 or y >= height:
        return 0
    return 1

def cut(frame, percentage=(0,1,0,1)):
    height = frame.shape[0]
    width = frame.shape[1]
    obj = frame[int(height*percentage[0]):int(height*percentage[1]),int(width*percentage[2]):int(width*percentage[3])]
    return obj

def dcut(frame, domain=(0,100,0,100)):
    """ domain: (yyxx)"""
    obj = frame[domain[0]:domain[1],domain[2]:domain[3]]
    return obj

"""debug global property"""
from control import pstatus
# pstatus = "release"
# pstatus = "debug"

"""设置全局错误处理"""
def handle_exception(exc_type, exc_value, exc_traceback):
    # 输出错误信息
    error_str = str(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print("发生了错误：", error_str)
    sys.exit(0)
    
def set_exit():
    
    if pstatus == "debug":
        sys.excepthook = handle_exception
        
def secdiff(t1: time.struct_time, t2: time.struct_time):
    t1 = t1[3]*3600 + t1[4]*60 + t1[5]
    t2 = t2[3]*3600 + t2[4]*60 + t2[5]
    return abs(t1 - t2)
        
def get_pid(p_start):
    ls = []
    pids = psutil.pids()
    for pid in pids:
        p = psutil.Process(pid)
        if p.name in 'python.exe':
            ls.append(p)
    p_t = time.strptime(p_start, "%H:%M:%S")
    p_obj = None
    p_mintime = 10000
    for each in ls:
        t = time.strptime(each[each.find('started=') + len('started=') + 1:-2], "%H:%M:%S")
        if secdiff(p_t, t) < p_mintime:
            p_mintime = secdiff(p_t, t)
            p_obj = p
    return p_obj
    
def upLift(pid):
    app = Application(backend='uia').connect(process=pid)
    we_chat_main_dialog = app.window(class_name='WeChatMainWndForPC')
    
    # 通过先最小化，再恢复使得窗口置顶
    we_chat_main_dialog.minimize()
    we_chat_main_dialog.restore()
    
if __name__ == '__main__':
    pass