# tkinterUI.py
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import cv2 as cv
from processing import *
from light import tractLight
from deal_data import Dealer
import utils
import sys, os
import control
from signals import msg_NoVideo

# 项目状态
pstatus = control.pstatus

# 提示信息
Prompt = "\n1.提取展示过程中按q退出\n2.按空格键暂停\n\n现在时间：{}\n欢迎使用二维目标追踪程序"

# 缩放比例 x: cm/px
View = 50 # 屏幕上50px -> 1cm
WIDTH = 1520
HEIGHT = 750
str_geometryProperty = f'{WIDTH}x{HEIGHT}+50+50'

class APP:
    """main window of APP
    """
    def on_step_change(self, *args):
        if self.step.get() == 'init':
            for each in [self.bt6,self.bt7,self.bt9, self.bt10, self.bt11, self.bt12,self.e1,self.ebt1,self.bt2,self.bt3,self.bt4,self.bt5,self.bt06]:
                each.config(state="disabled")
        elif self.step.get() == 'video loaded':
            for each in [self.bt6,self.bt10, self.bt12,self.e1,self.ebt1,self.bt2,self.bt3,self.bt4,self.bt5,self.bt06]:
                each.config(state="normal")
            for each in [self.bt7,self.bt11,self.bt9]:
                each.config(state="disabled")
        elif self.step.get() == 'light and process':
            self.result_txt.set('查看结果')
            for each in [self.bt7,self.bt11,self.bt9]:
                each.config(state="normal")
        elif self.step.get() == 'process without light':
            self.result_txt.set('查看结果(不带控制指令)')
            for each in [self.bt7,self.bt11,self.bt9]:
                each.config(state="normal")
        elif self.step.get() == 'all able':
            pass
        else:
            raise ValueError('wrong value')
    
    def __init__(self,master, pstatus) -> None:
        """
        Args:
            master: _description_
            pstatus: 'release' or 'debug'
            
        Properties:
            self.step: 定义一个被绑定的变量（发出信号）
        """
        self.project_status = pstatus
        self.master = master
        self.master.title('二维目标追踪视频处理程序')
        
        self.step = StringVar(master)
        self.step.trace("w", self.on_step_change)
        
        leftframe = Frame(master,width=40,height=40)
        leftframe.pack(side=LEFT,padx=10,pady=10)
        middleframe = Frame(master)
        middleframe.pack(side=LEFT,padx=10,pady=10)
        rightframe = Frame(master)
        rightframe.pack(side=RIGHT,padx=10,pady=10)
        
        # left frame: input and output control
        self.bt1 = Button(leftframe,width=25,height=3,text='导入视频',
                          font=("等线",20),bg='black',fg='white',command=self.load_video)
        self.bt1.pack(side=TOP)
        self.bt6 = Button(leftframe,width=25,height=3,text='提取闪光',
                          font=("等线",20),bg='black',fg='white',command=self.tract_light)
        self.bt6.pack(side=TOP)
        self.result_txt = StringVar(master)
        self.bt7 = Button(leftframe,width=25,height=3,textvariable=self.result_txt,
                          font=("等线",20),bg='green',fg='white',command=self.view_result)
        self.bt7.pack(side=TOP)
        self.result_txt.set('查看结果')
        self.bt8 = Button(leftframe,width=25,height=3,text='退出',
                          font=("等线",20),bg='orange',fg='black',command=self.quit)
        self.bt8.pack(side=TOP)
        
        # middle frame: display and process control
        self.lb1 = Label(middleframe, text='实验数据处理\n二维目标追踪系统',
                  font=("华文行楷",30),
                  padx=10,
                  pady=10
                  )
        self.lb1.pack(side=TOP)
        self.fps = '- '
        self.nframe = '- '
        self.video_width = '- '
        self.video_height = '- '
        self.lbconfig = Label(middleframe, text=f'num_frames : {self.nframe}, fps : {self.fps}', pady=5, font=('Times New Roman',15))
        self.lbconfig.pack(side=TOP)
        self.lbconfig_2 = Label(middleframe, text=f'width : {self.video_width}px, height : {self.video_height}px', pady=5, font=('Times New Roman',15))
        self.lbconfig_2.pack(side=TOP,pady=5)
        
        # peocess & display control
        self.bt10 = Button(middleframe, text='展示第一帧',pady=10, font=("等线",15,"underline"),relief=FLAT,command=self.show_first_frame)
        self.bt10.pack(side=TOP,pady=0)
        self.bt12 = Button(middleframe, text='跳帧读取',pady=10, font=("等线",15,"underline"),relief=FLAT,command=self.set_skip)
        self.bt12.pack(side=TOP,pady=0)
        self.bt11 = Button(middleframe, text='转换单位',pady=10, font=("等线",15,"underline"),relief=FLAT,command=self.go_change_unit)
        self.bt11.pack(side=TOP,pady=0)
        self.bt9 = Button(middleframe, text='取消转换单位',pady=10, font=("等线",15,"underline"),relief=FLAT,command=self.stop_change_unit)
        self.bt9.pack(side=TOP,pady=0)
        
        self.scale_textvar = StringVar(middleframe)
        self.lb2 = Label(middleframe, textvariable=self.scale_textvar,pady=5, font=('Times New Roman',15))
        self.lb2.pack()
        
        self.progressbar = ttk.Progressbar(middleframe,length=400)
        self.progressbar.pack(side=TOP,pady=20)
        self.progressbar['maximum'] = 100
        self.progressbar['value'] = 0
       
        self.hide_progressbar()
        # self.show_progressbar()
        
        # right frame: method choice
        self.bt2 = Button(rightframe,width=18,height=1,text='一般识别（前后点）',
                          font=("等线",15,"bold"),bg='blue',fg='white',activebackground='green',command=self.go_meanshift)
        self.bt2.pack(side=TOP,pady=10)
        self.bt3 = Button(rightframe,width=18,height=1,text='一般识别（整体）',
                          font=("等线",15,"bold"),bg='blue',fg='white',activebackground='green',command=self.go_camshift)
        self.bt3.pack(side=TOP,pady=10)
        self.bt4 = Button(rightframe,width=18,height=1,text='颜色识别',
                          font=("等线",15,"bold"),bg='blue',fg='white',activebackground='green',command=self.go_color)
        self.bt4.pack(side=TOP,pady=10)
        self.bt5 = Button(rightframe,width=18,height=1,text='标志点特征识别',
                          font=("等线",15,"bold"),bg='blue',fg='white',activebackground='green',command=self.go_feature)
        self.bt5.pack(side=TOP,pady=10)
        self.bt06 = Button(rightframe,width=18,height=1,text='边缘检测识别',
                          font=("等线",15,"bold"),bg='blue',fg='white',activebackground='green',command=self.go_contour)
        self.bt06.pack(side=TOP,pady=10)
        
        # bottom frame: process display
        bottomframe = Frame(rightframe,relief=SUNKEN)
        bottomframe.pack(side=BOTTOM,padx=10,pady=10)
        self.btr0 = Button(bottomframe,width=15,text='提取过程展示',
                          font=("等线",15,"bold"),bg='blue',fg='white',activebackground='green',command=self.go_display)
        self.btr0.pack(side=TOP,pady=(10,0))
        self.btr1 = Button(bottomframe,width=15,text='关闭提取过程展示',
                          font=("等线",15,"bold"),bg='red',fg='white',activebackground='green',command=self.stop_display)
        self.btr1.pack(side=TOP,pady=(0,10))

        # properties
        self.cap = None
        if(self.project_status == 'debug'):
            self.status = 'color'
            self.light = 1
            self.fps = 30
            self.filename = 'color_example.mp4'
            # self.magRatio = 0.0308*50
            self.magRatio = 0
            self.Ratio_to_cm = 0.0308
            self.timestr = utils.timestr()
            self.detect_mark_str = f'{self.status}-{self.timestr}'
            self.skip_num = 1
            self.step.set('light and process')
            self.scale_textvar.set('单位：px')
            self.data_unit = 'px'
            self.old_ratio = 1
        else:
            self.status = None
            self.light = 0
            self.fps = None
            self.filename = ''
            self.magRatio = 0
            self.Ratio_to_cm = 0
            self.timestr = None
            self.detect_mark_str = None
            self.skip_num = 1
            self.step.set('init')
            self.scale_textvar.set('单位：px')
            self.data_unit = 'px'
            self.old_ratio = 1
            
        self.output_window = None
        self.first_middle_point = (-1,-1)
        self.pm = 1
        self.rm = 1
        self.master.geometry(str_geometryProperty)
        self.tier2 = None
        
    def quit(self):
        sys.exit(0)
        
    def show_progressbar(self):
        self.progressbar.pack()

    def hide_progressbar(self):
        self.progressbar.pack_forget()
        
    def refresh(self):
        self.progressbar['value'] = 0
        self.master.update()
        
    def load_video(self):
        filepath = filedialog.askopenfilename(defaultextension='.mp4')
        if not filepath: # 打开文件失败
            # print('用户退出')
            return
        if not filepath.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.mpg', '.mpeg', '.flv', '.vob', '.3gp', '.3g2', '.asf', '.avchd', '.dv', '.matroska', '.mjpg', '.nsv', '.ogg', '.qt', '.riff', '.swf', '.videocd', '.webm')):
            showwarning('warning','请检查文件格式，不支持该格式文件')
            print('请检查文件格式，不支持该格式文件')
            return
        self.cap = cv.VideoCapture(filepath)
        print('',filepath,sep='\n')
        # 检测视频格式
        ret = self.cap.read()[0]
        if not ret:
            showwarning('warning','打开文件失败')
            print('打开文件失败')
            return
        self.filename = os.path.basename(filepath)
        self.video_width = int(self.cap.get(3))
        self.video_height = int(self.cap.get(4))
        self.fps = int(round(self.cap.get(5)))
        self.nframe = int(self.cap.get(7))
        self.lbconfig['text'] = f'帧数: {self.nframe}, 帧速率: {self.fps}帧/秒'
        self.lbconfig_2['text'] = f'宽度: {self.video_width}px, 高度: {self.video_height}px'
        # filename: absolute path
        self.light = 0
        self.status = None
        self.step.set('video loaded')
        self.master.update()
        
    def go_change_unit(self):
        if self.cap == None:
            showinfo(message=msg_NoVideo)
            return
        self.data_unit = 'cm'
        self.bodyLength, self.measure, self.first_middle_point = Magnify(self.cap, self.master)
        self.Ratio_to_cm = self.bodyLength / self.measure # 数据的缩放比例
        self.magRatio = View*self.bodyLength / self.measure # 显示的缩放比例
        if self.output_window and self.output_window.display:
            self.output_window.ratio = self.magRatio
        message = f'{self.bodyLength}cm : {self.measure:.1f}px \n or 1 px : {self.Ratio_to_cm:.4f}cm'
        showinfo(message=message)
        print('',message,sep='\n')
        self.scale_textvar.set('单位：cm')
        
    def stop_change_unit(self):
        self.data_unit = 'px'
        self.magRatio = 0
        if self.output_window and self.output_window.display:
            self.output_window.ratio = 0
        showinfo(message='已关闭缩放')
        self.scale_textvar.set('单位：px')
        
    def show_first_frame(self):
        if not self.cap:
            showinfo(message=msg_NoVideo)
            return
        self.cap.set(1, 0) # 重置为第一帧
        ret, frame0 = self.cap.read()
        cv.imwrite(f'.\\first_frame.png', frame0)
        my_show(frame0)
        
    def set_process_multiple(self):
        self.pm = eval(self.e1.get())
        if type(self.pm) not in [int, float]or self.pm <= 0:
            showinfo(message='请输入正整数！')
            self.pm = 1
        else:
            showinfo(message=f'已修改过程缩放倍数：{self.pm}')

    def set_reading_multiple(self):
        self.rm = eval(self.e2.get())
        if type(self.pm) not in [int, float] or self.rm <= 0:
            showinfo(message='请输入正数！')
            self.rm = 1
        else:
            showinfo(message=f'已修改过程缩放倍数：{self.rm}')
           
    def on_closing(self):
        self.output_window.display = 0
        self.output_window.close()
        
    def go_display(self):
        try:
            self.output_window.master.lift()
        except:
            self.tier2 = Tk()
            self.output_window = OutputWindow(self.tier2)
            self.output_window.display = 1
            if self.magRatio > 0:
                self.output_window.textboxprocess.insert('0.0','ratio: '+str(self.output_window.ratio)+'\n')
            self.output_window.startTime = utils.timestr()
            self.tier2.mainloop()
        
    def stop_display(self):
        if self.output_window is not None and self.output_window.display != 0:
            self.output_window.close()
            self.output_window.display = 0
        else:
            print('窗口已经关闭')
        
    def go_color(self):
        if self.cap==None :
            showinfo(message='请先导入文件')
            return
        self.show_progressbar()
        _rtn1 = main_color(self.cap,'front',self.master,
                   self.output_window,self.progressbar,self.pm,self.skip_num)
        
        _rtn2 = main_color(self.cap,'back',self.master,
                   self.output_window,self.progressbar,self.pm,self.skip_num)
        
        cv.destroyAllWindows()
        self.hide_progressbar()
        if _rtn1 != 'ok' or _rtn2 != 'ok':
            return
        if self.output_window and self.output_window.display:
            return
        
        self.status = 'color'
        if self.light:
            self.step.set('light and process')
        else:
            self.step.set('process without light')
        self.timestr = utils.timestr()
        self.detect_mark_str = f'{self.status}-{self.timestr}'
        
    def go_meanshift(self):
        if self.cap==None :
            showinfo(message='请先导入文件')
            return
        self.show_progressbar()
        showinfo(message='请选择前标记点矩形框')
        _rtn1 = meanshift(self.cap,'front',self.master,self.output_window,self.progressbar,self.pm,self.skip_num)
        if self.output_window and self.output_window.display:
            if _rtn1 == 'stop':
                self.output_window.textboxprocess.insert('0.0','提取过程中止\n')
            else:
                self.output_window.textboxprocess.insert('0.0','前点提取过程结束（展示过程不保存数据）\n')
        self.refresh()
        showinfo(message='请选择后标记点矩形框')
        _rtn2 = meanshift(self.cap,'back',self.master ,self.output_window,self.progressbar,self.pm,self.skip_num)
        if self.output_window and self.output_window.display:
            if _rtn2 == 'stop':
                self.output_window.textboxprocess.insert('0.0','提取过程中止\n')
            else:
                self.output_window.textboxprocess.insert('0.0','后点提取过程结束（展示过程不保存数据）\n')

        cv.destroyAllWindows()
        self.hide_progressbar()
        if _rtn1 != 'OK' or _rtn2 != 'OK':
            return
        if self.output_window and self.output_window.display:
            return
        
        self.status = 'meanshift'
        if self.light:
            self.step.set('light and process')
        else:
            self.step.set('process without light')
        self.timestr = utils.timestr()
        self.detect_mark_str = f'{self.status}-{self.timestr}'
        
    def go_contour(self):
        if self.cap==None :
            showinfo(message='请先导入文件')
            return
        self.show_progressbar()
        showinfo(message='请导入背景图')
        filepath = filedialog.askopenfilename(defaultextension='.jpg')
        if not self.is_image_path(filepath):
            showwarning('warning','未能识别的图片格式')
            print('未能识别的图片格式')
            return
        self.backgroundImg = cv.imread(filepath)
        _rtn = contour(self.cap,self.backgroundImg,self.master,self.output_window,self.progressbar,self.skip_num,use_contour=True)
        
        cv2.destroyAllWindows()
        self.hide_progressbar()
        if _rtn != 'OK':
            return
        if self.output_window and self.output_window.display:
            return
        
        self.status = 'contour'
        if self.light:
            self.step.set('light and process')
        else:
            self.step.set('process without light')
        self.timestr = utils.timestr()
        self.detect_mark_str = f'{self.status}-{self.timestr}'
        
    def go_camshift(self):
        if self.cap==None :
            showinfo(message='请先导入文件')
            return
        self.show_progressbar()
        '''camshift被包装在contour_camshift中'''
        _rtn = contour(self.cap,None,self.master,self.output_window,self.progressbar,self.skip_num,use_contour=False)
        
        cv.destroyAllWindows()
        self.hide_progressbar()
        if _rtn != 'OK':
            return
        if self.output_window and self.output_window.display:
            return
        
        self.status = 'camshift'
        if self.light:
            self.step.set('light and process')
        else:
            self.step.set('process without light')
        self.timestr = utils.timestr()
        self.detect_mark_str = f'{self.status}-{self.timestr}'

    def go_feature(self):
        if self.cap == None:
            showinfo(message=msg_NoVideo)
            return
        self.show_progressbar()
        showinfo(message='追踪前点')
        _rtn1 = feature(self.cap,kind='front',OutWindow=self.output_window,progressBar=self.progressbar, root=self.master, skip_n=self.skip_num)
        showinfo(message='追踪后点')
        _rtn2 = feature(self.cap,kind='back',OutWindow=self.output_window,progressBar=self.progressbar, root=self.master,skip_n=self.skip_num)
        
        cv.destroyAllWindows()
        self.hide_progressbar()
        if _rtn1 != 'OK' or _rtn2 != 'OK':
            return
        if self.output_window and self.output_window.display:
            return
        
        self.status = 'feature'
        if self.light:
            self.step.set('light and process')
        else:
            self.step.set('process without light')
        self.timestr = utils.timestr()
        self.detect_mark_str = f'{self.status}-{self.timestr}'
        
    def tract_light(self):
        if self.cap==None :
            showinfo(message='请先导入文件')
            return
        self.show_progressbar()
        showinfo(message='请点击灯的位置')
        rtn_ = tractLight(self.cap,self.master,self.output_window,self.progressbar)
        if rtn_ == 'OK':
            showinfo(message='闪光提取完成')
            self.light = 1
            if self.status is not None:
                self.step.set('light and process')
        else:
            if rtn_ == 'stop':
                showinfo(message='提取过程中止，展示模式不记录数据')
                self.output_window.textboxprocess.insert('0.0',"提取过程中止，展示模式不记录数据\n")
            else: # rtn_ == 'quit'
                pass
        cv.destroyAllWindows()
        self.hide_progressbar()
        self.refresh()
        
    def view_result(self):  
        data_dealer = Dealer(self.fps, f'{self.filename}({self.status})',self.master,self.progressbar,self.detect_mark_str,self.skip_num)
        self.dealer = data_dealer
        data_dealer.To_origin()
        if self.light:
            data_dealer.has_light = 1
            file_light = open('out-light-every.txt','r')
            data_dealer.parse_light(file_light, self.fps)
        else:
            data_dealer.has_light = 0
                
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
            data_dealer.parse_feature_result(file_f,file_b,self.fps)
            
        elif self.status == 'camshift':
            file_center = open('out-camshift-center.txt','r')
            file_angle = open('out-camshift-theta.txt','r')
            data_dealer.parse_center_angle(file_center,file_angle,self.fps)
            
        elif self.status == 'contour':
            file_center = open('out-contour-center.txt','r')
            file_angle = open('out-contour-theta.txt','r')
            data_dealer.parse_center_angle(file_center,file_angle,self.fps)
            
        else:
            showinfo(message='请先处理视频')
            return

        if self.data_unit == 'cm':
            data_dealer.data_change_ratio(self.Ratio_to_cm)
            data_dealer.To_centimeter(self.Ratio_to_cm)
            
        # create folders, if not exist
        dirs = ["results", # output_txt_directory
                "fig"] # output_png_directory
        for directory_name in dirs:
            if not os.path.exists(directory_name):
                os.mkdir(directory_name)
        
        self.show_result()
        
        tier1 = Tk()
        result_window = ResWindow(tier1,data_dealer)
        tier1.mainloop()
       
    def is_image_path(self, path):
        try:
            img = cv2.imread(path)
            if img is not None:
                return True
            else:
                return False
        except cv2.error:
            return False
        except FileNotFoundError:
            print("找不到该文件")
            return False
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def set_skip(self):
        self.dialog = tk.Toplevel(self.master)
        self.dialog.geometry("+600+300")
        self.dialog.title("输入框")
        self.dialog.resizable(False, False)
        tk.Label(self.dialog, text="请输入每（）帧记录一次：").pack(pady=10)
        self.entry_top = tk.Entry(self.dialog, width=20, font=("Consolas", 15))
        self.entry_top.pack(pady=10)
        tk.Button(self.dialog, text="确认", command=self.get_top_input).pack(pady=10)

    def get_top_input(self):
        input_text = self.entry_top.get()
        # print("输入的内容是：", input_text)
        try:
            self.skip_num = int(input_text)
        except:
            showinfo(message='请输入正整数')
        else:
            if self.skip_num <= 0:
                showinfo(message='请输入正整数')
                self.skip_num = 1
            else:
                showinfo(message='成功设置跳帧读取')
        finally:
            self.dialog.destroy()

    def show_result(self):
        showinfo(message='共提取到 %d帧信息，%d帧有效信息，%d次有效刺激' % 
        (min(self.dealer.num1,self.dealer.num2), self.dealer.num, len(self.dealer.stimulus)))
        print(f'\n有效信息:{self.dealer.num}\t 有效刺激:{len(self.dealer.stimulus)}\t 跳帧数:{self.skip_num}')
    
class ResWindow:
    def __init__(self,master,dealer) -> None:
        self.dealer = dealer
        self.master = master
        
        master.title('结果查看页面')
        master.geometry('420x375+600+400')
        
        button1 = Button(master, text='轨迹', width=20, font=('GB2312', 18), background='Tan', command=self.show_path)
        button1.grid(row=0, column=0, sticky=W)
        button2 = Button(master, text='角度', width=20, font=('GB2312', 18), background='Tan', command=self.show_angle)
        button2.grid(row=1, column=0, sticky=W)
        button02 = Button(master, text='曲率', width=20, font=('GB2312', 18), background='Tan', command=self.show_curve)
        button02.grid(row=2, column=0, sticky=W)
        button3 = Button(master, text='角速度和摆动角速度', width=20, font=('GB2312', 18), background='Tan', command=self.show_move)
        button3.grid(row=3, column=0, sticky=W)
        button03 = Button(master, text=' 精度：前后点距离变化 ', width=20, font=('GB2312', 18), background='Tan', command=self.show_dist)
        button03.grid(row=4, column=0, sticky=W)
        button4 = Button(master, text='返回', width=20, font=('GB2312', 18), background='Tan', command=master.destroy)
        button4.grid(row=5, column=0, sticky=W)
        
    def show_angle(self):
        self.dealer.showAngle()
        self.master.lift()   
        
    def show_path(self):
        self.dealer.showPath()
        self.master.lift()

    def show_curve(self):
        self.dealer.showCurve()
        self.master.lift()
        
    def show_move(self):
        self.dealer.showOmega()
        self.master.lift()
        
    def show_dist(self):
        self.dealer.showDist()
        self.master.lift()
        
class OutputWindow:
    def __init__(self,master) -> None:
        self.master = master
        
        master.title('过程显示页面')
        master.geometry('720x600+500+300')
        title = Label(master,text = "Output Board",font=('Bodoni MT',30),
                            anchor="center")
        title.pack(pady=10)
        
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
        self.textboxprocess.insert("insert","\n提示信息\n" + Prompt.format(utils.timestr()))
        self.startTime = ''
        self.display = 0
        self.ratio = 0
        
        # 关闭窗口协议
        master.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self.display = 0
        self.master.destroy()
        
    def close(self):
        self.display = 0
        self.master.destroy()

    def lift(self):
        self.master.lift()
        
    def WindowsLift(self):
        pid = utils.get_pid(self.startTime)
        utils.upLift(pid)
        
def main(ps):
    root = Tk()
    app = APP(root, ps)
    root.geometry(str_geometryProperty)
    root.mainloop()
    
def Reswindow_main():
    root = Tk()
    app = ResWindow(root, pstatus)
    root.mainloop()
