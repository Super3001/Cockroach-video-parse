# tkinter-test.py
import tkinter as tk
import sys

"""
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


import tkinter as tk

root = tk.Tk()

frame = tk.Frame(root)
frame.pack(padx = 10, pady = 10, fill="both", expand=True)
frame.propagate(False)

text = tk.Text(frame, font=("Arial", 12))
text.pack(fill="both", expand=True)

root.mainloop()
"""

'''import tkinter as tk

root = tk.Tk()
root.geometry("700x500")

# 创建标题
title = tk.Label(root, text="My Text Editor", font=("Arial", 20))
title.pack(pady=10)

# 创建可滚动的Text组件
canvas = tk.Canvas(root, bg="white", highlightthickness=0)
frame = tk.Frame(canvas, bg="white")
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)

frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=frame, anchor="nw")

text = tk.Text(frame, font=("Arial", 12))
text.pack(fill="both", expand=True, padx=10)

# 将组件放置在窗口中
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")
title.pack(side="top")

root.mainloop()'''

import tkinter as tk

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("输入框示例")
        
        # 创建一个按钮
        self.button = tk.Button(self.master, text="点击打开输入框", command=self.open_dialog)
        self.button.pack(pady=10)
        
        # 创建一个输入框
        self.entry = tk.Entry(self.master)
        
    def open_dialog(self):
        # 创建一个弹出窗口
        self.dialog = tk.Toplevel(self.master)
        self.dialog.title("输入框")
        
        # 在弹出窗口中添加一个输入框和一个确认按钮
        tk.Label(self.dialog, text="请输入内容：").pack(pady=10)
        self.entry.pack(pady=10)
        tk.Button(self.dialog, text="确认", command=self.get_input).pack(pady=10)
        
    def get_input(self):
        # 获取输入框中的内容
        input_text = self.entry.get()
        print("输入的内容是：", input_text)
        self.dialog.destroy()

root = tk.Tk()
app = App(root)
root.mainloop()
