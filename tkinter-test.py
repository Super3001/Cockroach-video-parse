# tkinter-test.py
import tkinter as tk
import sys

root = tk.Tk()
root.geometry("200x200")

# 创建三个Frame
frame1 = tk.Frame(root, bg="red", width=50, height=50)
frame2 = tk.Frame(root, bg="green", width=50, height=50)
frame3 = tk.Frame(root, bg="blue", width=50, height=50)

# 将它们打包并放置在主窗口中
frame1.pack(side=tk.LEFT)
frame2.pack(side=tk.LEFT)
frame3.pack(side=tk.LEFT)

def my_active():
    root.lift()

def new_window():
    new = tk.Tk()
    new.geometry("100x100")
    
    bt1 = tk.Button(new, text="active 0",command=my_active)
    bt1.pack(pady=10)
    new.mainloop()
    
def shutdown():
    sys.exit(0)

# 创建一个按钮，用于测试叠放顺序
button = tk.Button(root, text="new window", command=new_window)
button.pack(pady=10)

bt0 = tk.Button(root, text="close", command=shutdown)
bt0.pack(pady=10)

root.mainloop()
