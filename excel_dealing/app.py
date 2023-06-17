import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox as msg
from data_deal import *
import sys, os

class APP:
    def __init__(self, root) -> None:
        root.title('excel数据处理程序')
        window_width = 400
        window_height = 400
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x_coordinate = (screen_width - window_width) // 2
        y_coordinate = (screen_height - window_height) // 2
        root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_coordinate, y_coordinate))
        
        # 创建上面的按钮
        button1 = tk.Button(root, text="导入文件", height = 10, command=self.load_file)
        button1.pack(side="top", fill="x", expand=False)

        # 创建下面的按钮
        button2 = tk.Button(root, text="退出", height=5, command=self.quit)
        button2.pack(side="bottom", fill="both", expand=True)
        
        self.label1 = tk.Label(root, text="准备处理", height=5, font=("等线",15,"bold"), bg="black", fg="white")
        self.label1.pack(side="top", fill="x", expand=False)
        
        self.root = root
        self.dealer = None
        
    def load_file(self):
        filename = filedialog.askopenfilename(defaultextension='.xlsx')
        if not filename: # 打开文件失败
            msg.showinfo(message='打开文件失败')
            return
        dealer = Dealer(filename, self)
        self.dealer = dealer
        if(dealer.done):
            msg.showinfo(message=f'处理完成{dealer.done}页数据')
        else:
            msg.showwarning(message='处理异常')
        self.label1.config(text="准备处理")
        return
    
    def quit(self):
        self.root.destroy()
        
def main():   
    root = tk.Tk()
    app = APP(root)
    root.mainloop()
    flog.close()
    
if __name__ == '__main__':
    main()
        
    