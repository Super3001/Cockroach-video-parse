from tkinter import *
from tkinter import ttk
import tkinter.messagebox
from tkinter import filedialog
import cv2 as cv
from processing import *
from light import *
from deal_with_data import *
import utils
import sys, os
import control

# 项目状态
pstatus = control.pstatus

if pstatus == "debug":
    import PySimpleGUI as sg
    # sg.preview_all_look_and_feel_themes()
# 创建其他窗口并运行
# ...

# desktop = 'C:\\Users\\LENOVO\\Desktop\\'
图片 = '.\\src\\'

# 提示信息
Prompt = "\n1.图像展示过程按q退出\n2.按空格键暂停\n"
msg_NoVideo = '请先导入视频'

# 缩放比例 x: cm/px
View = 50 # 屏幕上50px -> 1cm
# 窗口大小
str_geometryProperty = '1600x850+50+50'

# 设置全局异常处理程序
# utils.set_exit()

class APP:
    def __init__(self,master) -> None:
        self.project_status = pstatus
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
        # temporary
        # self.lbp1 = Label(middleframe,image=self.lb1_photo)
        # self.lbp1.pack(side=TOP,padx=10,pady=20)
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
        if(self.project_status == 'debug'):
            self.status = 'meanshift'
            self.light = 1
            self.fps = 60
            self.filename = 'example.mp4'
            self.magRatio = 0.0308*50
            self.Ratio_to_cm = 0.0308
            self.timestr = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.detect_mark_str = f'{self.status}-{self.timestr}'
        else:
            self.status = None
            self.light = 0
            self.fps = None
            self.filename = ''
            self.magRatio = 0
            self.Ratio_to_cm = 0
            self.timestr = None
            self.detect_mark_str = None
        self.output_window = None
        self.first_middle_point = (-1,-1)
        self.pm = 1
        self.master.geometry('1200x700+50+50') # 失效？
        self.tier2 = None
        
    def quit(self):
        # self.master.destroy()
        sys.exit(0)
        
    def refresh(self):
        self.progressbar['value'] = 0
        self.master.update()
        
    def load_video(self):
        filename = filedialog.askopenfilename(defaultextension='.mp4')
        if not filename: # 打开文件失败
            return
        # tkinter.messagebox.showinfo(message=filename)
        self.cap = cv.VideoCapture(filename)
        if self.cap == None:
            return
        self.fps = int(round(self.cap.get(5)))
        self.nframe = int(self.cap.get(7))
        self.lbconfig['text'] = f'num_frames : {self.nframe}, fps : {self.fps}'
        # filename: absolute path
        pos = filename.rfind('/')
        self.filename = 'video'
        self.light = 0
        self.status = None
        self.master.update()
        
    def go_magnify(self):
        if self.cap == None:
            tkinter.messagebox.showinfo(message=msg_NoVideo)
            return
        self.bodyLength, self.measure, self.first_middle_point = Magnify(self.cap)
        self.Ratio_to_cm = self.bodyLength / self.measure
        self.magRatio = View*self.bodyLength / self.measure
        if self.output_window and self.output_window.display:
            self.output_window.ratio = self.magRatio
        tkinter.messagebox.showinfo(message=f'{self.bodyLength}cm : {self.measure:.1f}px \n'
                                    f'or 1 px : {self.Ratio_to_cm:.4f}cm')
        self.Ratio_to_m = self.magRatio*0.001
        # self.show_first_frame()
        self.master.geometry('1200x700+50+50')
        
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
        else:
            tkinter.messagebox.showinfo(message=f'已修改过程缩放倍数：{self.pm}')
           
    def on_closing(self):
        self.output_window.display = 0
        self.output_window.close()
        
    def go_display(self):
        try:
            self.output_window.master.lift()
        except:
            self.tier2 = Tk()
            # tier2.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.output_window = OutputWindow(self.tier2)
            self.output_window.display = 1
            if self.magRatio > 0:
                # self.output_window.ratio = self.magRatio
                self.output_window.textboxprocess.insert('0.0','ratio: '+str(self.output_window.ratio)+'\n')
            # tkinter.messagebox.showinfo(message='已打开提取过程展示')
            self.output_window.startTime = datetime.datetime.now().strftime('%H:%M:%S')
            print(self.output_window.startTime)
            self.tier2.mainloop()
        
    def stop_display(self):
        if self.output_window != None and self.output_window.display != 0 and self.output_window.master.winfo_exists(): 
            self.output_window.close()
            self.output_window.display = 0
            self.master.geometry(str_geometryProperty)
        # tkinter.messagebox.showinfo(message='已关闭提取过程展示')
        
    def go_color(self):
        if self.cap==None :
            tkinter.messagebox.showinfo(message='请先导入文件')
            return
        main_color(self.cap,self.master,self.output_window,self.progressbar)
        self.refresh()
        self.status = 'color'
        self.timestr = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.detect_mark_str = f'{self.status}-{self.timestr}'
        
    def go_meanshift(self):
        if self.cap==None :
            tkinter.messagebox.showinfo(message='请先导入文件')
            return
        tkinter.messagebox.showinfo(message='追踪前点，请拖动选择矩形框，然后回车')
        flag = meanshift(self.cap,'front',self.master,self.output_window,self.progressbar,self.pm)
        if self.output_window and self.output_window.display:
            if flag == 'stop':
                self.output_window.textboxprocess.insert('0.0','提取过程中止\n')
            else:
                self.output_window.textboxprocess.insert('0.0','前点提取过程结束（展示过程不保存数据）\n')
        self.refresh()
        tkinter.messagebox.showinfo(message='追踪后点，请拖动选择矩形框，然后回车')
        flag = meanshift(self.cap,'back',self.master ,self.output_window,self.progressbar,self.pm)
        if self.output_window and self.output_window.display:
            if flag == 'stop':
                self.output_window.textboxprocess.insert('0.0','提取过程中止\n')
            else:
                self.output_window.textboxprocess.insert('0.0','后点提取过程结束（展示过程不保存数据）\n')
        self.refresh()
        self.status = 'meanshift'
        self.timestr = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.detect_mark_str = f'{self.status}-{self.timestr}'
        
    def go_contour(self):
        if self.cap==None :
            tkinter.messagebox.showinfo(message='请先导入文件')
            return
        tkinter.messagebox.showinfo(message='请导入背景图')
        filename = filedialog.askopenfilename(defaultextension='.jpg')
        self.backgroundImg = cv.imread(filename)
        contour(self.cap,self.backgroundImg,self.master,self.output_window,self.progressbar)
        self.status = 'contour'
        self.timestr = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.detect_mark_str = f'{self.status}-{self.timestr}'
        
    def go_feature(self):
        if self.cap==None :
            tkinter.messagebox.showinfo(message='请先导入文件')
            return
        tkinter.messagebox.showinfo(message='追踪前点')
        feature(self.cap,kind='front',featureType='cross',OutWindow=self.output_window)
        tkinter.messagebox.showinfo(message='追踪后点')
        feature(self.cap,kind='back',featureType='cross',OutWindow=self.output_window)
        
        self.status = 'feature'
        self.timestr = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.detect_mark_str = f'{self.status}-{self.timestr}'
        
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
        data_dealer = Dealer(self.cap,self.filename,self.master,self.progressbar,self.detect_mark_str)
        data_dealer.To_origin()
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

        if self.magRatio > 0:
            data_dealer.data_change_ratio(self.Ratio_to_cm)
            data_dealer.To_centimeter(self.Ratio_to_cm)
        dirs = ["results", # output_txt_directory
                "fig"] # output_png_directory
        for directory_name in dirs:
            if not os.path.exists(directory_name):
                os.mkdir(directory_name)
        
        tier1 = Tk()
        result_window = ResWindow(tier1,data_dealer)
        tier1.mainloop()
        
    def go_help(self):
        # 放一个演示视频
        pass
    
class ResWindow:
    def __init__(self,master,dealer) -> None:
        self.dealer = dealer
        self.master = master
        
        master.title('结果查看页面')
        master.geometry('400x250+600+400')
        
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
        self.master.lift()   
        
    def show_path(self):
        self.show()
        self.dealer.showPath()
        self.dealer.showCurve()
        self.master.lift()
        
    def show_move(self):
        self.show()
        self.dealer.showOmega(self.dealer.fps)
        self.master.lift()
        
class OutputWindow:
    def __init__(self,master) -> None:
        self.master = master
        
        master.title('过程显示页面')
        master.geometry('720x600+500+300')
        title = Label(master,text = "Output Board",font=('Bodoni MT',30),
                            anchor="center")
        title.pack(pady=10)
        # lable_board.place(x = 10, y = 10)
        # 创建可滚动的Text组件
        canvas = tk.Canvas(master, bg="white", highlightthickness=0)
        frame = tk.Frame(canvas, bg="white")
        scrollbar = tk.Scrollbar(master, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frame, anchor="nw")

        text = tk.Text(frame, font=("Bodoni MT", 12))
        text.pack(fill="both", expand=True, padx=10)

        # 将组件放置在窗口中
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        title.pack(side="top")
        self.textboxprocess = text
        # self.textboxprocess.insert("insert","will be shown here\n")
        self.textboxprocess.insert("insert","\n提示信息\n" + Prompt)
        self.startTime = ''
        self.display = 0
        self.ratio = 0
        
    def close(self):
        self.master.destroy()

    def lift(self):
        self.master.lift()
        
    def WindowsLift(self):
        pid = utils.get_pid(self.startTime)
        utils.upLift(pid)
        
def main():   
    root = Tk()
    app = APP(root)
    root.geometry(str_geometryProperty)
    root.mainloop()
    
if __name__ == '__main__':
    main()