# debugUI.py
import tkinter as ttk
from tkinter import Tk
from importlib import reload

class myApp:

    def __init__(self):
        pass
        

    def _update(self):

        import tkinterUI

        self.main_refresh(tkinterUI)

    def main_refresh(self, python_script):

        reload(python_script)

        python_script.main()

def main():

    myapp = myApp()
    import tkinter as tk
    # 创建窗口
    window = tk.Tk()
    window.title("带按钮的窗口")
    # 创建按钮
    button = tk.Button(window, width=20,height=5, text="REFRESH",command=myapp._update)
    button.pack()
    # 进入主循环
    window.mainloop()

if __name__ == '__main__':
    main()