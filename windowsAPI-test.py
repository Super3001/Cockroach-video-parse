import psutil

def get_pid():
    pids = psutil.pids()
    for pid in pids:
        p = psutil.Process(pid)
        print(p)
        
get_pid()

def get_pid(p_start):
    pids = psutil.pids()
    for pid in pids:
        p = psutil.Process(pid)
        if p_start in str(p):
            return pid