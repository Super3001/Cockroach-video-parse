# utils.py 
import sys
import traceback

# pstate = "release"
pstate = "debug"

"""设置全局错误处理"""
def handle_exception(exc_type, exc_value, exc_traceback):
    # 输出错误信息
    error_str = str(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print("发生了错误：", error_str)
    sys.exit(0)
    
def set_exit():
    if pstate == "debug":
        sys.excepthook = handle_exception