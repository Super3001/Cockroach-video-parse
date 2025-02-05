from tkinter import *
from tkinter import ttk
import tkinter.messagebox
from tkinter import filedialog
import cv2 as cv
from processing import *
from light import *
from deal_with_data import *

# desktop = 'C:\\Users\\LENOVO\\Desktop\\'
图片 = '.\\src\\'

# 提示信息
Prompt = "\n1.图像展示过程按q退出\n"

class APP:
    def __init__(self,master) -> None:
        self.master = master
        self.master.title('蟑螂视频处理程序')
        leftframe = Frame(master,width=40,height=40)
        leftframe.pack(side=LEFT,padx=10,pady=10)
        middleframe = Frame(master)
        middleframe.pack(side=LEFT,padx=10,pady=10)
        rightframe = Frame(master)
        rightframe.pack(side=RIGHT,padx=10,pady=10)
        
        self.bt1 = Button(leftframe,width=25,height=3,text='导入视频',
                          font=("等线",20),bg='black',fg='white',command=self.load_video)
        self.bt1.pack(side=TOP)
        self.bt6 = Button(leftframe,width=25,height=3,text='提取闪光',
                          font=("等线",20),bg='black',fg='white',command=self.tract_light)
        self.bt6.pack(side=TOP)
        self.bt7 = Button(leftframe,width=25,height=3,text='查看结果',
                          font=("等线",20),bg='green',fg='white',command=self.view_result)
        self.bt7.pack(side=TOP)
        self.bt7 = Button(leftframe,width=25,height=3,text='退出',
                          font=("等线",20),bg='orange',fg='black',command=self.quit)
        self.bt7.pack(side=TOP)
        
        self.lb1 = Label(middleframe, text='实验数据处理\n带标记点的蟑螂追踪系统',  #文字支持换行
                  font=("华文行楷",30),
                  padx=10,
                  pady=10
                  )
        self.lb1.pack(side=TOP)
        self.lb1_photo = PhotoImage(file=图片+'background.png')
        self.fps = '-'
        self.nframe = '-'
        self.lbconfig = Label(middleframe, text=f'num_frames : {self.nframe}, fps : {self.fps}', pady=5, font=('Times New Roman',15))
        self.lbconfig.pack(side=TOP)
        self.lbp1 = Label(middleframe,image=self.lb1_photo)
        self.lbp1.pack(side=TOP,padx=10,pady=20)
        self.bt8 = Button(middleframe, text='视频缩放',pady=10, font=("等线",15,"underline"),relief=FLAT,command=self.go_magnify)
        self.bt8.pack(side=TOP,pady=0)
        self.bt9 = Button(middleframe, text='取消缩放',pady=10, font=("等线",15,"underline"),relief=FLAT,command=self.stop_magnify)
        self.bt9.pack(side=TOP,pady=0)
        middledownframe = Frame(middleframe)
        middledownframe.pack(side=TOP,padx=10,pady=10)
        self.lb2 = Label(middledownframe,text='处理的缩放倍数：',font=("等线",15))
        self.lb2.grid(row=1,column=1)
        self.e1 = Entry(middledownframe, font=("等线",15),relief=FLAT,width=12)
        self.e1.insert(0, "1(请输入整数)")
        self.e1.grid(row=1,column=2)
        self.ebt1 = Button(middledownframe, text='确定', font=("等线",15,"underline"),relief=FLAT,command=self.set_process_multiple)
        self.ebt1.grid(row=1,column=3)
        self.progressbar = ttk.Progressbar(middleframe,length=200)
        self.progressbar.pack(side=TOP,pady=20)
        self.progressbar['maximum'] = 100
        self.progressbar['value'] = 0
        self.bt10 = Button(middleframe, text='展示第一帧',pady=10, font=("等线",15,"underline"),relief=FLAT,command=self.show_first_frame)
        self.bt10.pack(side=TOP,pady=0)
        self.bt2 = Button(rightframe,width=15,height=1,text='meanshift',
                          font=("等线",15,"bold"),bg='blue',fg='white',activebackground='green',command=self.go_meanshift)
        self.bt2.pack(side=TOP,pady=10)
        self.bt3 = Button(rightframe,width=15,height=1,text='颜色识别',
                          font=("等线",15,"bold"),bg='blue',fg='white',activebackground='green',command=self.go_color)
        self.bt3.pack(side=TOP,pady=10)
        self.bt4 = Button(rightframe,width=15,height=1,text='轮廓识别',
                          font=("等线",15,"bold"),bg='blue',fg='white',activebackground='green',command=self.go_contour)
        self.bt4.pack(side=TOP,pady=10)
        self.bt5 = Button(rightframe,width=15,height=1,text='标志点特征识别',
                          font=("等线",15,"bold"),bg='blue',fg='white',activebackground='green',command=self.go_feature)
        self.bt5.pack(side=TOP,pady=10)
        
        bottomframe = Frame(rightframe,relief=SUNKEN)
        bottomframe.pack(side=BOTTOM,padx=10,pady=10)
        
        self.bt8 = Button(bottomframe,width=15,text='提取过程展示',
                          font=("等线",15,"bold"),bg='blue',fg='white',activebackground='green',command=self.go_display)
        self.bt8.pack(side=TOP,pady=(10,0))
        self.bt9 = Button(bottomframe,width=15,text='关闭提取过程展示',
                          font=("等线",15,"bold"),bg='red',fg='white',activebackground='green',command=self.stop_display)
        self.bt9.pack(side=TOP,pady=(0,10))
        self.cap = None
        # self.status = None
        self.status = 'meanshift'
        # self.light = 0
        self.light = 1
        self.fps = 60
        self.output_window = None
        self.magRatio = 0
        self.first_middle_point = (-1,-1)
        self.pm = 1
        self.master.geometry('1200x600+100+100')
        
    def quit(self):
        self.master.destroy()
        
    def refresh(self):
        self.progressbar['value'] = 0
        self.master.update()
        
    def load_video(self):
        filename = filedialog.askopenfilename(defaultextension='.mp4')
        # tkinter.messagebox.showinfo(message=filename)
        self.cap = cv.VideoCapture(filename)
        if self.cap == None:
            return
        self.fps = int(round(self.cap.get(5)))
        self.nframe = int(self.cap.get(7))
        self.lbconfig['text'] = f'num_frames : {self.nframe}, fps : {self.fps}'
        self.master.update()
        
    def go_magnify(self):
        self.bodyLength, self.measure, self.first_middle_point = Magnify(self.cap)
        self.magRatio = 50*self.bodyLength / self.measure
        if self.output_window and self.output_window.display:
            self.output_window.ratio = self.magRatio
        # self.show_first_frame()
        
    def stop_magnify(self):
        self.magRatio = 0
        if self.output_window and self.output_window.display:
            self.output_window.ratio = 0
        tkinter.messagebox.showinfo(message='已关闭缩放')
        
    def show_first_frame(self):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        ret, frame0 = self.cap.read()
        print('ratio:',self.magRatio)
        my_show(frame0,self.magRatio,self.first_middle_point)
        
    def set_process_multiple(self):
        self.pm = eval(self.e1.get())
        if type(self.pm) != int or self.pm <= 0:
            tkinter.messagebox.showinfo(message='请输入正整数！')
            self.pm = 1
        pass
        
    def go_display(self):
        if self.output_window == None or self.output_window.display == 0: 
            tier2 = Tk()
            self.output_window = OutputWindow(tier2)
            self.output_window.display = 1
            if self.magRatio > 0:
                self.output_window.ratio = self.magRatio
                print('ratio:',self.output_window.ratio)
            # tkinter.messagebox.showinfo(message='已打开提取过程展示')
            tier2.mainloop()
        else:
            # tkinter.messagebox.showinfo(message='已打开提取过程展示')
            pass
        
    def stop_display(self):
        if self.output_window != None and self.output_window.display != 0 and self.output_window.master.winfo_exists(): 
            self.output_window.close()
            self.output_window.display = 0
            self.master.geometry('1200x600+100+100')
        # tkinter.messagebox.showinfo(message='已关闭提取过程展示')
        
    def go_color(self):
        if self.cap==None :
            tkinter.messagebox.showinfo(message='请先导入文件')
            return
        main_color(self.cap,self.master,self.output_window,self.progressbar)
        self.refresh()
        self.status = 'color'
        
    def go_meanshift(self):
        if self.cap==None :
            tkinter.messagebox.showinfo(message='请先导入文件')
            return
        tkinter.messagebox.showinfo(message='追踪前点，请点击左上角和右下角，然后回车')
        meanshift(self.cap,'front',self.master,self.output_window,self.progressbar,self.pm)
        self.refresh()
        tkinter.messagebox.showinfo(message='追踪后点，请点击左上角和右下角，然后回车')
        meanshift(self.cap,'back',self.master ,self.output_window,self.progressbar,self.pm)
        self.refresh()
        self.status = 'meanshift'
        
    def go_contour(self):
        if self.cap==None :
            tkinter.messagebox.showinfo(message='请先导入文件')
            return
        tkinter.messagebox.showinfo(message='请导入背景图')
        filename = filedialog.askopenfilename(defaultextension='.jpg')
        self.backgroundImg = cv.imread(filename)
        contour(self.cap,self.backgroundImg,self.master,self.output_window,self.progressbar)
        self.status = 'contour'
        
    def go_feature(self):
        if self.cap==None :
            tkinter.messagebox.showinfo(message='请先导入文件')
            return
        tkinter.messagebox.showinfo(message='追踪前点')
        feature(self.cap,kind='front',featureType='cross',OutWindow=self.output_window)
        tkinter.messagebox.showinfo(message='追踪后点')
        feature(self.cap,kind='back',featureType='cross',OutWindow=self.output_window)
        
        self.status = 'feature'
        
    def tract_light(self):
        if self.cap==None :
            tkinter.messagebox.showinfo(message='请先导入文件')
            return
        tkinter.messagebox.showinfo(message='请点击灯的位置，然后回车')
        tractLight(self.cap,self.master,self.output_window,self.progressbar)
        tkinter.messagebox.showinfo(message='闪光提取完成')
        self.light = 1
        self.refresh()
        
    def view_result(self):
        data_dealer = Dealer(self.cap,self.master,self.progressbar)
        if self.light:
            file_light = open('out-light-every.txt','r')
            data_dealer.deal_time(file_light, self.fps)
        else:
            tkinter.messagebox.showinfo(message='请先提取闪光')
            return
                
        if self.status == 'color':
            file_f = open('out-color-1.txt','r')
            file_b = open('out-color-2.txt','r')
            data_dealer.parse_fbpoints(file_f,file_b,self.fps)
        
        elif self.status == 'meanshift':
            file_f = open('out-meanshift-1.txt','r')
            file_b = open('out-meanshift-2.txt','r')
            data_dealer.parse_fbpoints(file_f,file_b,self.fps)
            
        elif self.status == 'feature':
            file_f = open('out-feature-1.txt','r')
            file_b = open('out-feature-2.txt','r')
            data_dealer.parse_fbpoints(file_f,file_b,self.fps)
            
        elif self.status == 'contour':
            file_center = open('out-contour-center.txt','r')
            file_angle = open('out-contour-theta.txt','r')
            data_dealer.parse_center_angle(file_center,file_angle,self.fps)
            
        else:
            tkinter.messagebox.showinfo(message='请先处理视频')
            return
    
        tier1 = Tk()
        result_window = ResWindow(tier1,data_dealer)
        tier1.mainloop()
        
    def go_help(self):
        # 放一个演示视频
        pass
        
    
class ResWindow:
    def __init__(self,master,dealer) -> None:
        self.dealer = dealer
        
        
        master.title('结果查看页面')
        master.geometry('400x200+600+400')
        
        button1 = Button(master, text='轨迹和转向半径', width=20, font=('GB2312', 18), background='Tan', command=self.show_path)
        button1.grid(row=0, column=0, sticky=W)
        button2 = Button(master, text='角度', width=20, font=('GB2312', 18), background='Tan', command=self.show_angle)
        button2.grid(row=1, column=0, sticky=W)
        button3 = Button(master, text='角速度和摆动角速度', width=20, font=('GB2312', 18), background='Tan', command=self.show_move)
        button3.grid(row=2, column=0, sticky=W)
        button4 = Button(master, text='返回', width=20, font=('GB2312', 18), background='Tan', command=master.destroy)
        button4.grid(row=3, column=0, sticky=W)
        
        # self.show()
        
    def show(self):
        tkinter.messagebox.showinfo(message='共提取到%d帧信息，%d次有效刺激' % (len(self.dealer.Theta),len(self.dealer.lighttime)))

    def show_angle(self):
        self.show()
        self.dealer.showAngle(self.dealer.fps)
        
        
    def show_path(self):
        self.show()
        self.dealer.showPath()
        self.dealer.showCurve()
        
    def show_move(self):
        self.show()
        self.dealer.showOmega(self.dealer.fps)
        
class OutputWindow:
    def __init__(self,master) -> None:
        self.master = master
        master.title('过程显示页面')
        master.geometry('500x400+600+400')
        lable_board = Label(master,text = "Output Board",font=('Bodoni MT',30))
        lable_board.place(x = 100, y = 10)
        self.textboxprocess = Text(master)
        self.textboxprocess.place(x=10,y=80,width=500,height=300)
        self.textboxprocess.insert("insert","will be shown here\n")
        self.textboxprocess.insert("insert","\n提示信息\n" + Prompt)
        
        self.display = 0
        self.ratio = 0
        
    def close(self):
        self.master.destroy()

root = Tk()
app = APP(root)

root.mainloop()