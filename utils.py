# utils.py 
import sys
import traceback
import psutil
from pywinauto.application import Application
import time

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
    
