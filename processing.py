import cv2 as cv
from tkinter.messagebox import showinfo, showerror 
from tract_point import *
import numpy as np
import math
import time
from alive_progress import alive_bar
from utils import dcut
from sklearn.cluster import KMeans

"""debug global property"""
from control import pstatus
# pstatus == "release"
# pstatus == "debug"

from utils import Stdout_progressbar

SHOW_TIME = 100 # 默认的展示间隔时间，默认为0.1s，单位：ms

# def my_show(frame, ratio=1, center_point=(-1,-1), _time=0):
def my_show(frame, ratio=1, _time=0):
    """
    frame: 要展示的帧
    ratio: 缩放比例
    _time: 显示时间
    
    """
    # print(center_point)
    h = frame.shape[0]
    w = frame.shape[1]
    frame = cv.resize(frame, (int(w*ratio), int(h*ratio)))
    cv.imshow("window", frame)
    key = cv.waitKey(_time)
    if key == ord('q'):
        return 1
    elif key == 32: # 空格键暂停
        if _time>0:
            return my_show(frame, ratio, _time=0)
        else:
            pass
    return 0

""""""
def restrict_to_boundary(domain, h, w):
    """
        domain: yyxx
    """
    return (
        max(domain[0], 0),
        min(domain[1], h),
        max(domain[2], 0),
        min(domain[3], w)
    )

def printb(s, OutWindow, p=False): # 打印到output board上
    s = str(s)
    OutWindow.textboxprocess.insert("0.0",s+'\n')
    if p:
        print(s)

def expand(frame, mutiple=1):
    # print('执行了这个函数')
    height, width, channel = frame.shape
    return cv.resize(frame,(round(width*mutiple), round(height*mutiple)))

def dist(A, B):
    return math.sqrt((B[0] - A[0])**2 + (B[1] - A[1])**2)

def rect_cover1(frame, thres_value=0, edge=0): # 在指定的图像中用一个矩形覆盖所有值大于thres_value的点
    upper = (frame.shape[0],frame.shape[1])
    lower = (-1,-1)
    left = (frame.shape[0],frame.shape[1])
    right = (-1,-1)
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if frame[i][j] > thres_value:
                if i > lower[0]:
                    lower = (i,j)
                if i < upper[0]:
                    upper = (i,j)
                if j > right[1]:
                    right = (i,j)
                if j < left[1]:
                    left = (i,j)
    return ((upper[0]-edge,lower[0]+edge,left[1]-edge,right[1]+edge), 
            (tuple(reversed(upper)), tuple(reversed(lower)), tuple(reversed(left)), tuple(reversed(right)))) # 上，下，左，右，先y后x

def count_points(frame,thres=1): # 灰度图
    points = []
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if frame[i][j]>=thres:
                points.append((j,i))
    return points

def cleanout(points, domain):
    X = [x[0] for x in points]
    Y = [x[1] for x in points]
    
    # 中位数过滤
    X = list(sorted(X))
    Y = list(sorted(Y))
    xmid = X[len(X)//2] if len(X)%2 == 1 else (X[len(X)//2-1] + X[len(X)//2]) / 2
    ymid = Y[len(Y)//2] if len(Y)%2 == 1 else (Y[len(Y)//2-1] + Y[len(Y)//2]) / 2
    
    cleaned_points = []
    for p in points:
        if abs(p[0]-xmid) > domain[0] or abs(p[1]-ymid) > domain[1]:
            continue
        cleaned_points.append(p)
        
    return cleaned_points

def print_mid_point(rect):
    if len(rect) == 4:
        x = (rect[2]+rect[3])/2
        y = (rect[0]+rect[1])/2
        return str(round(x,2))+','+str(round(y,2))
    elif len(rect) == 2:
        return str(round(rect[0],2))+','+str(round(rect[1],2))
    else:
        raise Exception('rect的长度错误，应为2或4')

def extract_features_from_mask(mask, pre_point, thresh_dist):
    # 寻找所有为True的点的坐标
    points = np.argwhere(mask)
    
    # 计算每个点与前置点的距离
    distances = np.linalg.norm(points - pre_point, axis=1)
    
    # 找到距离小于等于阈值的点的索引
    valid_indices = np.where(distances <= thresh_dist)[0]
    
    # 提取符合条件的点的中心位置
    valid_points = points[valid_indices]
    center_positions = valid_points.mean(axis=0)
    
    return center_positions

def color_deal(frame,midval:tuple,dis:int,pre_point,thresh_dist,OutWindow=None):
    """
    用颜色过滤
    frame: 截取过后的处理帧
    midval: 颜色
    dis: 颜色的范围
    pre_point: 上一次提取的center
    thresh_dist: 阈值
    OutWindow: 展示窗

    return value: 
        center_point: (y, x)
    """
    # low_bound=tuple(map(lambda x: int(x-dis),midval))
    # high_bound=tuple(map(lambda x: int(x+dis),midval))
    low_bound = np.array(midval) - dis
    high_bound = np.array(midval) + dis

    mask = cv.inRange(frame,low_bound,high_bound)
    points = np.argwhere(mask)
    if len(points) == 0:
        return 0,0
    
    if np.all(pre_point == -1): # 第一帧或者前一帧为black
        '''用kmeans算法进行聚类，选择包含点最多的类的中心位置作为center'''
        kmeans = KMeans(n_clusters=3, n_init='auto') # 聚类的数量为3
        kmeans.fit(points)

        # 获取每个点所属的类别
        labels = kmeans.labels_

        # 可视化聚类结果
        image_show = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        for i, each in enumerate(points):
            if labels[i] == 0:
                cv2.circle(image_show, (each[1], each[0]), 2, (0, 0, 255), -1)
            elif labels[i] == 1:
                cv2.circle(image_show, (each[1], each[0]), 2, (0, 255, 0), -1)
            else:
                cv2.circle(image_show, (each[1], each[0]), 2, (255, 0, 0), -1)
        
        for center in kmeans.cluster_centers_:
            cv2.circle(image_show, (int(center[1]), int(center[0])), 2, (255, 255, 255), -1)
        cv2.imshow("image_show",image_show)
        cv2.waitKey(0)

        # 如果中心点的距离小于阈值，则认为是同一个点
        center_positions = kmeans.cluster_centers_

        # 计算每个中心点到其他中心点的距离
        distances = np.linalg.norm(center_positions - center_positions[:, np.newaxis], axis=2)

        linked = distances < thresh_dist

        ''' linked相当于所有中心点形成的无向图的邻接矩阵，下面需要计算每个强连通分量的中心点
            用kasaraju算法计算强连通分量'''
        # 计算邻接矩阵的转置
        linked_transpose = linked.T

        # 从任意一个点开始进行深度优先搜索，得到搜索的顺序
        visited = np.zeros(linked.shape[0], dtype=bool)
        point_color = np.zeros(linked.shape[0], dtype=np.int32)
        order = []
        sccCnt = [0] # 为了在函数中可以修改外部变量
        
        def dfs(i):
            visited[i] = True
            for j in range(linked.shape[0]):
                if linked[i, j] and not visited[j]:
                    dfs(j)
            order.append(i)

        def dfs_transpose(i):
            point_color[i] = sccCnt[0]
            for j in range(linked.shape[0]):
                if linked_transpose[i, j] and point_color[j] == 0:
                    dfs_transpose(j)

        def kasaraju():
            for i in range(linked.shape[0]):
                if not visited[i]:
                    dfs(i)

            # visited[:] = False
            for i in reversed(order):
                if point_color[i] == 0:
                    sccCnt[0] = sccCnt[0] + 1
                    dfs_transpose(i)

        # 计算强连通分量
        kasaraju()
        
        # 统计出scc
        group_id = np.unique(point_color)
        scc = np.zeros((len(group_id),), dtype=np.object_)
        for i, each in enumerate(group_id):
            scc[i] = np.where(point_color == each)[0]
        
        print('scc',scc)

        '''# 根据强连通分量合并中心点
        # for each in scc:
            # center_positions[each] = center_positions[each].mean(axis=0)'''

        # 根据强连通分量合并聚类
        for each in scc:
            center_positions[each] = center_positions[each].mean(axis=0)
        
        # 把每一类的label统一为这个类中最小的label、
        for each in scc:
            idx = [i for i, label in enumerate(labels) if label in each]
            labels[idx] = each.min()

        # 统计每个类别中的点的数量
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(unique_labels, counts)

        # 找到包含点最多的类的类别号
        max_count_index = unique_labels[np.argmax(counts)]

        # 获取包含点最多的类的中心位置
        center_point = center_positions[max_count_index].astype(int)[::-1]

        # 可视化center_point

    else:
        '''到前置点的距离小于阈值的点的中心位置作为center'''
        # 计算每个点与前置点的距离
        distances = np.linalg.norm(points - pre_point, axis=1)
        
        # 找到距离小于等于阈值的点的索引
        valid_indices = np.where(distances <= thresh_dist)[0]
        
        # 提取符合条件的点的中心位置(平均/外接矩形中心)
        valid_points = points[valid_indices]
        center_point = valid_points.mean(axis=0).astype(int)[::-1]


    '''center_point: (x, y)'''
    # print(center_point)
    border = 10
    x_left = center_point[0] - border
    x_right = center_point[0] + border
    y_up = center_point[1] - border
    y_down = center_point[1] + border

    '''    # all_rect, points = rect_cover1(mask,0,10)
    X = count_points(mask)
    
    X_final = cleanout(X,(200,200))
    if len(X_final)<10:
        return 0,0
    x_left=min([x[0] for x in X_final])
    x_right=max([x[0] for x in X_final])
    y_up=min([y[1] for y in X_final])
    y_down=max([y[1] for y in X_final])
    midpoint=((x_left+x_right)//2 , (y_up+y_down)//2)'''
    
    if OutWindow and OutWindow.display:
        frame_show1 = cv.rectangle(frame,(x_left, y_up),(x_right,y_down),(0,0,255),1)
        frame_show1 = cv.circle(frame_show1,center_point,1,color=(0,0,255))
        if my_show(frame_show1, ratio=1, _time=0):
            return 'q',0
    
    return 'ok', center_point[::-1] # center_point: (y, x)

"""颜色识别的主函数"""
def main_color(cap,root,OutWindow,progressBar,pm=1,skip_n=1):  # 颜色提取
    border = 20
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    ret, frame0 = cap.read()
    if not ret:
        showerror(message='读入错误')
        return
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), 
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))) 
    num = int(cap.get(7))
    # Trc = Tractor()
    # Trc.set("mutiple",pm)
    # showinfo('','请提取前点颜色，确定按回车')
    # midval_f = Trc.tract_color(frame0)
    midval_f = [136, 136, 170]
    if midval_f[0] == -1:
        return 'stop'
    if OutWindow and OutWindow.display:
        OutWindow.textboxprocess.delete('0.0','end')
        OutWindow.textboxprocess.insert('0.0','前点颜色(BGR)'+str(midval_f))
        OutWindow.textboxprocess.insert('0.0',"帧序号：[中心点坐标]\n")
    # showinfo(message='前点颜色(BGR)'+str(midval_f)+' ...请等待')
    
    cap.set(cv.CAP_PROP_POS_FRAMES, 0) # 重置为第一帧
    domain = (0,frame0.shape[0],0,frame0.shape[1]) # 上下左右
    domain = [int(x*pm) for x in domain]
    success = 1
    cnt = 0
    pre_point = np.array([-1, -1])
    progressBar['maximum'] = num*2
    file = open('out-color-1.txt','w')
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    with alive_bar(math.ceil(num/skip_n)) as bar:
        while success:
            success, frame = cap.read()
            if not success:
                break
            cnt += 1
            frame = expand(frame,pm)
            if skip_n > 1 and cnt % skip_n != 1:
                continue
            progressBar['value'] = cnt
            root.update()

            rtn, center = color_deal(frame[domain[0]:domain[1]+1,domain[2]:domain[3]+1],midval_f, 15, pre_point, np.linalg.norm(size) // 5, OutWindow)
            # thresh_dist = np.linalg.norm(size) // 5
            
            if rtn==0:
                pre_point = (-1, -1)
                domain = (0,size[1] - 1,0,size[0] - 1)
                if OutWindow and OutWindow.display:
                    printb(str(cnt)+': '+'black', OutWindow)
                else:
                    file.write(f'{cnt} 0,0\n')
            elif rtn=='q':
                printb('用户退出', OutWindow)
                cv.destroyAllWindows()
                break
            else:
                frame_show = frame.copy()
                domain = (domain[0]+center[0]-border,domain[0]+center[0]+border, domain[2]+center[1]-border,domain[2]+center[1]+border)
                # restrict to boundary
                domain = restrict_to_boundary(domain, size[1], size[0])

                if OutWindow and OutWindow.display:
                    frame_show = cv.circle(frame_show, center, 2, (0,0,255))
                    if cnt == 1:
                        if my_show(dcut(frame_show, domain),ratio=4, _time=100):
                            cv.destroyAllWindows()
                            break
                    else:
                        if my_show(dcut(frame_show, domain),ratio=4, _time=100):
                            cv.destroyAllWindows()
                            break
                    printb(str(cnt)+': '+print_mid_point(center), OutWindow)
                else:
                    file.write(f'{cnt} {print_mid_point(center)}\n') # standard format

                pre_point = center

            bar()
        # while
        
        


    file.close()
    
    showinfo('', '请提取后点颜色，确定按回车')
    midval_b = Trc.tract_color(frame0)    
    if midval_b[0] == -1:
        return 'stop'
    if OutWindow and OutWindow.display:
        OutWindow.textboxprocess.delete('0.0','end')
        OutWindow.textboxprocess.insert('0.0','后点颜色(BGR)'+str(midval_b))
        OutWindow.textboxprocess.insert('0.0',"帧序号：[中心点坐标]\n")
    showinfo(message='后点颜色(BGR)'+str(midval_b)+' ...请等待')
    
    cap.set(cv.CAP_PROP_POS_FRAMES, 0) # 重置为第一帧
    domain = (0,frame0.shape[0],0,frame0.shape[1]) # 上下左右
    success = 1
    cnt = 0
    file = open('out-color-2.txt','w')
    with alive_bar(math.ceil(num/skip_n)) as bar:
        while success:
            success, frame = cap.read()
            if not success:
                break
            cnt += 1
            frame = expand(frame,pm)
            if skip_n > 1 and cnt % skip_n != 1:
                continue
            progressBar['value'] = cnt + num
            root.update()

            domain = [max(x,0) for x in domain]
            rtn, rect = color_deal(frame[domain[0]:domain[1]+1,domain[2]:domain[3]+1],list(midval_b),15, 0)
            if rtn==0:
                domain = (0,frame0.shape[0],0,frame0.shape[1])
                if OutWindow and OutWindow.display:
                    OutWindow.textboxprocess.insert('0.0',str(cnt)+': '+'black\n')
                else:
                    file.write(f'{cnt} 0,0\n')
            elif rtn=='q':
                break
            else:
                domain = (rect[0]+domain[0]-border,rect[1]+domain[0]+border, rect[2]+domain[2]-border,rect[3]+domain[2]+border)
                if OutWindow and OutWindow.display:
                    frame_show = frame.copy()
                    frame_show = cv.circle(frame_show, middle_point(domain), 2, (0,0,255))
                    if my_show(frame_show):
                        break
                    if my_show(dcut(frame_show, domain)):
                        break
                    OutWindow.textboxprocess.insert('0.0',str(cnt)+': '+print_mid_point(domain) + '\n')
                else:
                    file.write(f'{cnt} {print_mid_point(domain)}\n') # standard format
            bar()

    
    file.close()
    cv.destroyAllWindows() 
    if OutWindow and OutWindow.display:
        pass
    else:
        showinfo(message='检测完成！')
    pass

"""meanshift方法识别主函数"""
def meanshift(cap,kind,root=None,OutWindow=None,progressBar=None,pm=1, skip_n=1):
    # 输入进度条所在窗口，输出窗口，进度条和处理缩放比率
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), 
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    num = cap.get(7)
    Trc = Tractor()
    Trc.set('mutiple',pm)
    r, h, c, w = Trc.select_rect(frame)
    if r == None:
        return 'stop'
    track_window = (c, r, w, h)
    # print(track_window)
    frame = expand(frame,pm)
    roi = frame[r:r+h, c:c+w]

    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    # 3.3 计算直方图
    roi_hist = cv.calcHist([hsv_roi], [0], None, [180], [0, 180])

    # 3.4 归一化
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

    # 4. 目标追踪
    # 4.1 设置窗口搜索终止条件：最大迭代次数，窗口中心漂移最小值
    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

    cap.set(cv.CAP_PROP_POS_FRAMES, 0) # 重置为第一帧
    
    stdoutpb = Stdout_progressbar(num, not(OutWindow and OutWindow.display))
    cnt = 0
    progressBar['maximum'] = num
    # OutWindow.discontinue = False
    # print(OutWindow.ratio)
    
    if kind == 'front':
        if OutWindow and OutWindow.display:
            OutWindow.lift()
            # OutWindow.WindowsLift()
            # OutWindow.textboxprocess.delete('0.0','end')
            OutWindow.textboxprocess.insert('0.0',"帧序号：[中心点坐标]\n")
        else:
            file = open('out-meanshift-1.txt','w')
        stdoutpb.reset(skip_n)
        while(True):
            # 4.2 获取每一帧图像
            
            ret, frame = cap.read()
            if ret == True:
                cnt += 1
                if skip_n > 1 and cnt % skip_n != 1:
                    continue
                progressBar['value'] = cnt
                root.update()

                frame = expand(frame,pm)
                x, y, w, h = track_window
                if OutWindow and OutWindow.display:
                    OutWindow.textboxprocess.insert("0.0",str(cnt) + ': [' + print_mid_point((y, y+h, x, x+w)) + ']\n')
                else:
                    file.write(f'{cnt} {print_mid_point((y, y+h, x, x+w))}\n') # standard format
                    # file.write(print_mid_point((y, y+h, x, x+w)) + '\n')

                # 4.3 计算直方图的反向投影
                hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

                # 4.4 进行meanshift追踪
                ret, track_window = cv.meanShift(dst, track_window, term_crit)

                # 4.5 将追踪的位置绘制在视频上，并进行显示
                if OutWindow and OutWindow.display:
                    x, y, w, h = track_window
                    img2 = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
                    
                    rtn = my_show(img2,OutWindow.ratio,midPoint((x, y), (x + w, y + h)), 60)
                    if rtn == 1:
                        return 'stop'
                
                stdoutpb.update(cnt)
            else:
                break
    else:
        if OutWindow and OutWindow.display:
            OutWindow.lift()
            # OutWindow.textboxprocess.delete('0.0','end')
            OutWindow.textboxprocess.insert('0.0',"帧序号：[中心点坐标]\n")
        else:
            file = open('out-meanshift-2.txt','w')
        stdoutpb.reset(skip_n)
        while(True):
            # 4.2 获取每一帧图像
            
            ret, frame = cap.read()
            if ret == True:
                cnt += 1
                if skip_n > 1 and cnt % skip_n != 1:
                    continue
                progressBar['value'] = cnt
                root.update()

                frame = expand(frame,pm)
                x, y, w, h = track_window
                if OutWindow and OutWindow.display:
                    OutWindow.textboxprocess.insert("0.0",str(cnt) + ': [' + print_mid_point((y, y+h, x, x+w)) + ']\n')
                else:
                    file.write(f'{cnt} {print_mid_point((y, y+h, x, x+w))}\n') # standard format

                # 4.3 计算直方图的反向投影
                hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

                # 4.4 进行meanshift追踪
                ret, track_window = cv.meanShift(dst, track_window, term_crit)

                # 4.5 将追踪的位置绘制在视频上，并进行显示
                if OutWindow and OutWindow.display:
                    x, y, w, h = track_window
                    img2 = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
                    
                    if my_show(img2,OutWindow.ratio,midPoint((x, y), (x + w, y + h)), 60):
                        return 'stop'
                
                stdoutpb.update(cnt)
            else:
                break
    cv.destroyAllWindows()
    if OutWindow and OutWindow.display:
        return 'OK'
    file.close()
    showinfo(message='检测完成！')
    return 'OK'

def midPoint(A,B):
    x1,y1 = A
    x2,y2 = B
    return ((x1+x2)/2 , (y1+y2)/2)

def middle_point(rect:list):
    x = (rect[2]+rect[3])//2
    y = (rect[0]+rect[1])//2
    return (y,x)
    
def Magnify(cap, root):
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    ret, frame0 = cap.read()
    Trc = Tractor()
    showinfo(message='请点击蟑螂前端，然后回车')
    Trc.tractPoint(frame0)
    point_f = Trc.gbPoint
    showinfo(message='请点击蟑螂后端，然后回车')
    Trc.tractPoint(frame0)
    point_b = Trc.gbPoint
    length = dist(point_f, point_b)
    Trc.inputbox(root=root,show_text='请输入蟑螂的实际长度，单位：厘米')
    body_length = eval(Trc.gbInput)
    # my_show(frame0, 50*body_length/length, midPoint(point_b,point_f))
    return body_length, length, midPoint(point_b,point_f)

'''之前写的旋转函数，现在用cv2的函数实现'''
def rotate(img, angle, center=None, scale=1.0):
    """
        param:
            img: the img to be rotated
            angle: the angle to rotate the img (in degree)
            center: the center of the img
            scale: the scale of the img
    """
    # Determine the center of the image
    (h, w) = img.shape[:2]
    if center is None:
        center = (w/2, h/2)
    
    # Define the rotation matrix
    M = cv.getRotationMatrix2D(center, angle, scale)

    # Apply the rotation to the image
    rotated = cv.warpAffine(img, M, (w, h)) 
    return rotated

"""计算一个点绕另一个点旋转后的坐标"""
def rotate_point(point, center, deg):
    # 将角度转换为弧度
    rad = math.radians(deg)

    # 提取点的坐标
    x, y = point
    cx, cy = center

    # 计算相对于中心点的坐标
    rel_x = x - cx
    rel_y = y - cy

    # 应用旋转矩阵
    rotated_x = rel_x * math.cos(rad) - rel_y * math.sin(rad)
    rotated_y = rel_x * math.sin(rad) + rel_y * math.cos(rad)

    # 计算旋转后的点的坐标
    new_x = rotated_x + cx
    new_y = rotated_y + cy

    return new_x, new_y

'''卷积操作结果，用cv2的函数实现'''
def conv2d_res(frame, kernal, pos, angle=0):
    # 将图片逆时针旋转angle角度
    rows, cols = frame.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1) 
    rotated = cv2.warpAffine(frame, M, (cols, rows)) 
    frame = rotated

    # 对应的点也转过angle角度
    pos = rotate_point(pos, (cols/2, rows/2), angle)

    return cv.filter2D(frame, cv.CV_16S, kernal)[pos[0]][pos[1]] # cv.CV_16S代表输出的数据类型为16位整数
    
    """手动实现"""
    # h,w = kernal.shape
    # x,y = pos
    # res = 0
    # for i in range(h):
        # for j in range(w):
            # res += frame[x+i][y+j]*kernal[i][j]
    # print(res)
    # return res

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
        if my_show(frame_show, _time=100):
            return (-1,-1), 0
        print('max_value: ' + str(max_value))
        print('max_pos: ' + str(max_pos_ori))
    return max_pos_ori, max_value

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

def edge(frame):
    
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

def get_frame(number):
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    while number:
        rtn, frame = cap.read()
        number-=1
    my_show(frame)
    cv.imwrite("C:\\Users\\LENOVO\\Desktop\\obj_frame.jpg",frame)
        
def rect_cover(frame, thres_value=0): # 在指定的图像中用一个矩形覆盖所有值大于thres_value的点
    upper = (frame.shape[0],frame.shape[1])
    lower = (-1,-1)
    left = (frame.shape[0],frame.shape[1])
    right = (-1,-1)
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if frame[i][j] > thres_value:
                if i > lower[0]:
                    lower = (i,j)
                if i < upper[0]:
                    upper = (i,j)
                if j > right[1]:
                    right = (i,j)
                if j < left[1]:
                    left = (i,j)
    return ((upper[0],lower[0],left[1],right[1]), 
            (tuple(reversed(upper)), tuple(reversed(lower)), tuple(reversed(left)), tuple(reversed(right)))) # 上，下，左，右，先y后x

def rect_points(points):
    X = [x[0] for x in points]
    Y = [y[1] for y in points]
    return (min(Y), max(Y), min(X), max(X))

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
    showinfo(message='请拖动选择初始矩形框，之后回车')
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
        
def cleanout(points, rect):
    X = [x[0] for x in points]
    Y = [x[1] for x in points]
    
    # 中位数过滤
    X = list(sorted(X))
    Y = list(sorted(Y))
    xmid = X[len(X)//2] if len(X)%2 == 1 else (X[len(X)//2-1] + X[len(X)//2]) / 2
    ymid = Y[len(Y)//2] if len(Y)%2 == 1 else (Y[len(Y)//2-1] + Y[len(Y)//2]) / 2
    
    cleaned_points = []
    for p in points:
        if abs(p[0]-xmid) > rect[0] or abs(p[1]-ymid) > rect[1]:
            continue
        cleaned_points.append(p)
        
    # print('mid:',xmid,ymid)
    # print('origin:',len(points),'now:',len(cleaned_points))
    
    return cleaned_points
    
def tilt(edge, display):
    rect0, minmax_points = rect_cover(edge,55)
    # if display:
    #     edge_show0 = edge.copy()
    #     edge_show0 = cv.rectangle(edge_show0, (rect0[2],rect0[0]), (rect0[3],rect0[1]), 255, 2)
    #     if my_show(edge_show0):
    #         return (None,) * 3
    # 用处理后的线性回归方法做角度计算
    points = count_points(edge,55)
    if len(points) < 10: # 采集到的点太少
        return (0,0,0,0), 0, 0
    points = cleanout(points,(100,100))
    
    rect = rect_points(points)
    if display:
        edge_show1 = edge.copy()
        edge_show1 = cv.rectangle(edge_show1, (rect[2],rect[0]), (rect[3],rect[1]), 255, 2)
        print(rect0)
        print(rect)
    
    X = [x[0] for x in points]
    Y = [-x[1] for x in points]
    f = np.polyfit(X,Y,1)
    angle = math.atan(f[0])*180/math.pi
    if display:
        print('angle:',angle)
        # edge_show1 = cv.circle(edge_show1,(rect[2]-5,(rect[0]+rect[1])//2+int(f[0]*((rect[3]-rect[2])/2+5))),3,255)
        # edge_show1 = cv.circle(edge_show1,(rect[3]+5,(rect[0]+rect[1])//2-int(f[0]*((rect[3]-rect[2])/2+5))),3,255)
        edge_show1 = cv.line(edge_show1, (rect[2]-5,(rect[0]+rect[1])//2+int(f[0]*((rect[3]-rect[2])/2+5))), (rect[3]+5,(rect[0]+rect[1])//2-int(f[0]*((rect[3]-rect[2])/2+5))), 255, 2)
        if my_show(edge_show1):
            cv2.destroyAllWindows()
            return (None,) * 3
    return rect, angle, len(points)

"""轮廓方法识别主函数"""
def contour(cap,background,root,OutWindow,progressBar, skip_n=1, turn_start=0,turn_end=0): # 根据外轮廓算角度
    
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    Trc = Tractor()
    ret, frame0 = cap.read()
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), 
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    frame_num = cap.get(7)
    showinfo(message='请拖动选择初始矩形框，之后回车')
    file_theta = open('out-contour-theta.txt','w')
    file_center = open('out-contour-center.txt','w')
    r, h, c, w = Trc.select_rect(frame0)
    if r == None:
        return 'stop'
    domain = (r,r+h,c,c+w) # 上下左右
    cha = cv.subtract(edge(dcut(frame0,domain)),edge(dcut(background,domain)))
    # cha = cv.inRange(cha,120,255)
    if OutWindow and OutWindow.display:
        if my_show(cha):
            return 'stop'
    # harris(cha,dcut(frame0,domain))
    rect, angle, num = tilt(cha, OutWindow and OutWindow.display)
    if not rect:
        return 'stop'
    border = 40
    if num < 10:
        domain = (0, frame0.shape[0], 0, frame0.shape[1])
        # file_center.write('0, 0\n')
        # file_theta.write('0\n')
    else:
        domain = (rect[0]+domain[0]-border,rect[1]+domain[0]+border, rect[2]+domain[2]-border,rect[3]+domain[2]+border)
        if domain[0] < 0:
            domain[0] = 0
        if domain[2] < 0:
            domain[2] = 0
        if OutWindow and OutWindow.display:
            printb(f'angle: {angle}', OutWindow)
        # file_theta.write(str(round(angle,2))+'\n')  
        # file_center.write(print_mid_point(domain)+'\n')  
    stdoutpb = Stdout_progressbar(frame_num, not(OutWindow and OutWindow.display))
    progressBar['maximum'] = frame_num
    success = 1
    cnt = 0
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    stdoutpb.reset(skip_n)
    while success:
        success, frame = cap.read()
        if not success:
            break
        cnt += 1
        if skip_n > 1 and cnt % skip_n != 1:
            continue
        progressBar['value'] = cnt
        root.update()

        cha = cv.subtract(edge(dcut(frame,domain)),edge(dcut(background,domain)))
        if OutWindow and OutWindow.display:
            OutWindow.textboxprocess.insert('0.0', f'{cnt}:\n')
        if cnt < turn_start:
            continue
        if turn_end > 0 and cnt > turn_end:
            break
        
        rect, angle, num = tilt(cha, OutWindow and OutWindow.display)
        if not rect:
            return 'stop'

        if num < 10:
            domain = (0, frame0.shape[0], 0, frame0.shape[1])
            if OutWindow and OutWindow.display:
                OutWindow.textboxprocess.insert('0.0', f'(0, 0)  0\n')
            else:
                file_center.write(f'{cnt} 0,0\n')
                file_theta.write(f'{cnt} 0\n')
            continue
        domain = (rect[0]+domain[0]-border,rect[1]+domain[0]+border, rect[2]+domain[2]-border,rect[3]+domain[2]+border)
        if domain[0] < 0:
            domain[0] = 0
        if domain[2] < 0:
            domain[2] = 0
        if OutWindow and OutWindow.display:
            OutWindow.textboxprocess.insert('0.0', f'({print_mid_point(domain)})  {round(angle,2)}\n')
        else:
            file_center.write(f'{cnt} {print_mid_point(domain)}\n')
            file_theta.write(f'{cnt} {round(angle,2)}\n')
        stdoutpb.update(cnt)
        
    if OutWindow and OutWindow.display:
        OutWindow.textboxprocess.insert('检测完成，展示模式不修改数据\n')
    else:
        file_center.close()
        file_theta.close()
        showinfo(message='检测完成!')
    return 'OK'

class FakeMs:
    def __init__(self) -> None:
        self.cnt = 0
    
    def update(self):
        self.cnt += 1

if pstatus == "debug":
    cap = cv2.VideoCapture(r"D:\GitHub\Cockroach-video-parse\src\DSC_2059.MOV")
    ret, frame0 = cap.read()
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    from tkinter import *
    Prompt = "this is a debug trial "
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
            
    if __name__ == '__main__':
        tier = Tk()
        window = OutputWindow(tier)
        window.display = 1
        window.textboxprocess.insert("0.0", "111\n")
        # main_color(cap,root=FakeMs(),OutWindow=None,progressBar=dict(),skip_n=10)
        main_color(cap,root=FakeMs(),OutWindow=window,progressBar=dict(),skip_n=10)
        tier.mainloop()
