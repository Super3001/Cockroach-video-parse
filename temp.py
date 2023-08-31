import cv2 as cv
import numpy
import math

def max_conv2d(frame, domain, kernal, angle, display):
    """
    frame: 原图像
    domain: 卷积作用域
    kernal: 卷积核
    angle: 旋转角度
    display: 是否显示结果
    """

    # 转为弧度
    rad = angle * math.pi / 180

    # 旋转图像
    rows, cols = frame.shape
    M = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated = cv.warpAffine(frame, M, (cols, rows))
    frame = rotated

    max_value = -1e7
    max_pos = (-1,-1)
    for x in range(domain[0],domain[1]+1):
        for y in range(domain[2],domain[3]+1):
            # 旋转对应点
            x1 = int((x - cols/2) * math.cos(rad) - (y - rows/2) * math.sin(rad) + cols/2)
            y1 = int((x - cols/2) * math.sin(rad) + (y - rows/2) * math.cos(rad) + rows/2)
            # 计算卷积
            res = cv.filter2D(frame, cv.CV_16S, kernal)[x1, y1]
            # frame[y1][x1] = numpy.sum(frame[y-domain[2]:y+domain[3]+1, x-domain[0]:x+domain[1]+1] * kernal)         
            # 更新最大值
            if res > max_value:
                max_value = res
                max_pos = (x,y)

    # 计算最大值对应的中心点
    max_pos_ori = max_pos
    h, w = kernal.shape
    max_pos_center = (max_pos[0] + h//2, max_pos[1] + w//2)

    if display:
        frame_show = cv.cvtColor(frame,cv.COLOR_GRAY2BGR)
        frame_show = cv.circle(frame_show,tuple(reversed(max_pos_center)),3,(0,0,255))
        frame_show = dcut(frame_show,(max_pos_center[0]-20,max_pos_center[0]+1+20,max_pos_center[1]-20,max_pos_center[1]+1+20))
        if pstatus == 'debug':
            # print(frame_show.shape)
            pass
        if my_show(frame_show, time=100):
            return (-1,-1), 0
        print('max_value: ' + str(max_value))
        print('max_pos: ' + str(max_pos_ori))
    return max_pos_ori, max_value

def feature(cap,kind='front',OutWindow=None,progressBar=None,root=None, skip_n=1):
    offset = 5
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    frame_num = cap.get(7) # 获取视频总帧数

    ret, frame0 = cap.read() # 读取第一帧
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), 
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))) # 获取视频尺寸
    
    Idf = Identifier()
    showinfo(message='请拖动选择初始矩形框，之后回车')
    rtn_ = Idf.select_window(frame0)
    if rtn_ == 'q': # 用户取消
        printb('', OutWindow)
        return
    
    (x,y,w,h), minis = rtn_ # minis未启用
    # 如果是展示模式，输出到窗口，否则打开文件句柄
    if OutWindow and OutWindow.display:
        printb("0 :", OutWindow)
    else:
        file = open('out-feature-1.txt','w') if kind == 'front' else open('out-feature-2.txt','w')

    domain = (y - offset,y + offset,x - offset,x + offset)
    max_angle = 0 # 初始的angle必须是0
    max_angle_pos = max_conv2d(cv.cvtColor(frame0, cv.COLOR_BGR2GRAY),domain,Idf.K, 0, OutWindow and OutWindow.display)[0]
    domain = (max_angle_pos[0]-offset,max_angle_pos[0]+offset, max_angle_pos[1]-offset,max_angle_pos[1]+offset) # 更新domain

    # 初始化参数
    stdoutpb = Stdout_progressbar(frame_num, not(OutWindow and OutWindow.display))
    progressBar['maximum'] = frame_num
    success = 1
    cnt = 0
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    stdoutpb.reset(skip_n=skip_n)

    while success:
        success, frame = cap.read()
        if not success:
            stdoutpb.update(-1)
            break
        cnt += 1
        if skip_n > 1 and cnt % skip_n != 1:
            continue
        progressBar['value'] = cnt
        root.update()
        
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if OutWindow and OutWindow.display:
            printb(f'{cnt} :', OutWindow)
        max_value = 0
        max_angle = pre_angle
        for angle in range(pre_angle-5,pre_angle+5+1): # 循环11次
            pos, value = max_conv2d(frame,domain,Idf.K, angle, OutWindow and OutWindow.display)
            if pos == (-1,-1): # abnormal status
                print('end(manually)')
                cv.destroyAllWindows()
                return 1
            if value > max_value:
                max_value = value
                max_angle_pos = pos
                max_angle = angle
        
        domain = (max_angle_pos[0]-12,max_angle_pos[0]+12, max_angle_pos[1]-12,max_angle_pos[1]+12) # 更新domain
        if OutWindow and OutWindow.display:
            printb('now_pos:' + str(max_angle_pos), OutWindow)
            printb('max_angle:' + str(max_angle), OutWindow)
        else:
            file.write(f'{cnt} {max_angle_pos[0]},{max_angle_pos[1]}\n') # standard format
        pre_angle = max_angle
        stdoutpb.update(cnt)
        
    # 结束
    showinfo(message='检测完成！')
    if OutWindow and OutWindow.display:
        cv.destroyAllWindows()
    else:
        file.close()
    return 0
    
    
