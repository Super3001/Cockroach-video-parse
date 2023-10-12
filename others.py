# others.py
import numpy as np

K_cross = np.array([
    [-4,-4,-4,5,5,5,-4,-4,-4],
    [-4,-4,-4,5,5,5,-4,-4,-4],
    [-4,-4,-4,5,5,5,-4,-4,-4],
    [5,5,5,-4,-4,-4,5,5,5],
    [5,5,5,-4,-4,-4,5,5,5],
    [5,5,5,-4,-4,-4,5,5,5],
    [-4,-4,-4,5,5,5,-4,-4,-4],
    [-4,-4,-4,5,5,5,-4,-4,-4],
    [-4,-4,-4,5,5,5,-4,-4,-4]
])

K_add = np.array([
    [5,5,5,-4,-4,-4,5,5,5],
    [5,5,5,-4,-4,-4,5,5,5],
    [5,5,5,-4,-4,-4,5,5,5],
    [-4,-4,-4,-4,-4,-4,-4,-4,-4],
    [-4,-4,-4,-4,-4,-4,-4,-4,-4],
    [-4,-4,-4,-4,-4,-4,-4,-4,-4],
    [5,5,5,-4,-4,-4,5,5,5],
    [5,5,5,-4,-4,-4,5,5,5],
    [5,5,5,-4,-4,-4,5,5,5]
])

K_rect = np.array([
    [-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,8,8,8,-1,-1,-1],
    [-1,-1,-1,8,8,8,-1,-1,-1],
    [-1,-1,-1,8,8,8,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1]
])

'''previous version of max_conv2d''' 
'''def max_conv2d(frame, domain, K, display=1, now_angle=0, rotate_angle=5):
    """
        param:
            frame: the img to be convoluted
            domain: the domain of the frame to be convoluted
            K: the kernal
            display: whether to display the result
            rotate_angle: the angle to rotate the img
    """

    # transform the angle to rad
    now_rad = np.deg2rad(now_angle)
    rotate_rad = np.deg2rad(rotate_angle)

    # rotate the kernal to previous position
    K = rotate(K,now_rad)

    # rotate the kernal clockwise by rotate_angle
    K_cw = rotate(K,rotate_rad)

    # rotate the kernal counterclockwise by rotate_angle
    K_ccw = rotate(K,-rotate_rad)

    max_value = -1e7
    max_pos = (-1,-1)
    max_angle = 0
    for idx, _K in enumerate([K_cw,K,K_ccw]):
        for i in range(domain[0],domain[1]+1):
            for j in range(domain[2],domain[3]+1):
                now_value = conv2d_res(frame,_K,(i,j))
                if now_value > max_value:
                    max_value = now_value
                    max_pos = (i,j)
                    max_angle = (idx-1)*rotate_angle
                    
    if max_value == 0:
        return (-1,-1)
    
    max_pos_ori = max_pos
    h, w = K.shape
    max_pos_center = (max_pos[0] + h//2, max_pos[1] + w//2)

    if display:
        frame_show = cv.cvtColor(frame,cv.COLOR_GRAY2BGR)
        frame_show = cv.circle(frame_show,tuple(reversed(max_pos_center)),3,(0,0,255))
        frame_show = dcut(frame_show,(max_pos_center[0]-20,max_pos_center[0]+1+20,max_pos_center[1]-20,max_pos_center[1]+1+20))
        if pstatus == 'debug':
            # print(frame_show.shape)
            pass
        if my_show(frame_show, _time=100):
            return (-1,-1), 0
        print('max_value: ' + str(max_value))
        print('max_pos: ' + str(max_pos))
        print('max_angle: ' + str(max_angle))
    return max_pos_ori, now_angle+max_angle
'''

"""previous version of edge"""
'''def edge(frame):
    
    frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    robert = (kernelx, kernely)
    prewitt = (np.array([[-1,-1,-1],[0,0,0],[1,1,1]]),
               np.array([[-1,0,1],[-1,0,1],[-1,0,1]]))
    
    x = cv.filter2D(frame, cv.CV_16S, prewitt[0])
    y = cv.filter2D(frame, cv.CV_16S, prewitt[1])
    
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    res = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    return res
'''

"""
def contour(frame):  # 利用内置方法找到轮廓
    res = frame.copy()
    frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    frame = cv.GaussianBlur(frame, (11, 11), 0)
    
    edge = cv.Canny(frame,30,150)
    my_show(edge)
    
    contour, rtn = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(res, contour,-1,(0,255,0),2)
    my_show(res)
    return res
"""

"""previous version: convolution方法识别主函数"""
'''def feature(cap,kind='front',OutWindow=None,progressBar=None,root=None, skip_n=1): 
    offset = 5
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    frame_num = cap.get(7)
    
    ret, frame0 = cap.read()
    print(frame0.shape)
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), 
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    Idf = Identifier()
    showinfo(message='请选择初始矩形框')
    rtn_ = Idf.select_window(frame0)
    if rtn_ == 'q':
        printb('', OutWindow)
        return
    (x,y,w,h), minis = rtn_
    if OutWindow and OutWindow.display:
        printb("0 :", OutWindow)
    else:
        file = open('out-feature-1.txt','w') if kind == 'front' else open('out-feature-2.txt','w')

    domain = (y,y+h,x,x+w)
    # domain[0] = max(0, domain[0] - offset)
    # domain[1] = min(size[1], domain[1] + offset)
    # domain[2] = max(0, domain[2] - offset)
    # domain[3] = min(size[0], domain[3] + offset)
    now_pos, now_angle = max_conv2d(cv.cvtColor(frame0, cv.COLOR_BGR2GRAY),domain,Idf.K, OutWindow and OutWindow.display, 0, 0)

    domain = (now_pos[0]-offset,now_pos[0]+offset, now_pos[1]-offset,now_pos[1]+offset)
    
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

        # rotate the frame
        frame = rotate(frame, now_angle)

        if OutWindow and OutWindow.display:
            printb(f'{cnt} :', OutWindow)
        now_pos, now_angle = max_conv2d(frame,domain,Idf.K, OutWindow and OutWindow.display, now_angle)
        if now_pos == (-1,-1): # abnormal status
            print('end(manually)')
            cv.destroyAllWindows()
            return 1

        domain = (now_pos[0]-12,now_pos[0]+12, now_pos[1]-12,now_pos[1]+12)
        if OutWindow and OutWindow.display:
            printb('now_pos:' + str(now_pos), OutWindow)
            printb('now_angle:' + str(now_angle), OutWindow)
        else:
            # file.write(f'{cnt} {str(now_pos).replace("(","").replace(")","")}\n')
            file.write(f'{cnt} {now_pos[0]},{now_pos[1]}\n') # standard format

        stdoutpb.update(cnt)
        
    showinfo(message='检测完成！')
    file.close()
    cv.destroyAllWindows()
    return 0'''
