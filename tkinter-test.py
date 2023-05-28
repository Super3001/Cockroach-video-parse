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
import tkinter as tk

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

root.mainloop()
