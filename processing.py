import cv2 as cv
from tkinter.messagebox import showinfo, showerror
from tract_point import *
import numpy as np
import math
from alive_progress import alive_bar
from utils import dcut
from sklearn.cluster import KMeans

"""debug global property"""
from control import pstatus
# pstatus == "release"
# pstatus == "debug"

from utils import Stdout_progressbar

SHOW_TIME = 100 # 默认的展示间隔时间，默认为0.1s，单位：ms
MAX_VALUE_THRESH = 1000
OUTER_BORDER = 200

class BGRimg:
    '''
        3 channels for [b, g, r] in range (0, 255)
    '''

class Grayimg:
    '''
        one channel gray img
    '''

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
    if ratio == 0:
        frame = cv.resize(frame, (1200, 800))
    else:
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
        domain: `y,y,x,x`
    """
    return (
        max(domain[0], 0),
        min(domain[1], h - 1),
        max(domain[2], 0),
        min(domain[3], w - 1)
    )

def in_boundary(point, h, w):
    """
        point: (y, x)
    """
    return point[0] >= 0 and point[0] < h and point[1] >= 0 and point[1] < w

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

def print_mid_point(rect: list|tuple, sep=','):
    if len(rect) == 4:
        x = (rect[2]+rect[3])/2
        y = (rect[0]+rect[1])/2
        return str(round(x,2))+sep+str(round(y,2))
    elif len(rect) == 2:
        return str(round(rect[0],2))+sep+str(round(rect[1],2))
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

def color_deal(frame,midval:tuple,dis:int,pre_state,thresh_dist,OutWindow=None):
    """
    用颜色过滤
    frame: 截取过后的处理帧
    midval: 颜色
    dis: 颜色的范围
    pre_state: 上一帧的状态，是否为black
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
    # least_points = 20
    if len(points) < 20: 
        return 0, 0  # 0, 0表示没有在范围内的点（或者符合要求的点太少）

    # 如果前一帧为black，需要重写找点
    if pre_state == -1: # 第一帧或者前一帧为black
        '''用kmeans算法进行聚类，选择包含点最多的类的中心位置作为center'''
        kmeans = KMeans(n_clusters=3, n_init='auto') # 聚类的数量为3
        kmeans.fit(points)

        # 获取每个点所属的类别
        labels = kmeans.labels_

        # 可视化聚类结果
        # img_show = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        img_show = frame.copy()
        for i, each in enumerate(points):
            if labels[i] == 0:
                cv2.circle(img_show, (each[1], each[0]), 2, (0, 0, 255), -1)
            elif labels[i] == 1:
                cv2.circle(img_show, (each[1], each[0]), 2, (0, 255, 0), -1)
            else:
                cv2.circle(img_show, (each[1], each[0]), 2, (255, 0, 0), -1)
        
        for center in kmeans.cluster_centers_:
            cv2.circle(img_show, (int(center[1]), int(center[0])), 2, (255, 255, 255), -1)
        # cv2.imshow("img_show",img_show)
        # cv2.waitKey(0)

        # 如果中心点的距离小于阈值，则认为是同一个点
        center_positions = kmeans.cluster_centers_

        # 计算每个中心点到其他中心点的距离
        distances = np.linalg.norm(center_positions - center_positions[:, np.newaxis], axis=2)

        linked = distances < thresh_dist

        ''' linked相当于所有中心点形成的无向图的邻接矩阵，下面需要计算每个强连通分量(团？)的中心点
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
        
        print('cluster scc',scc)

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
        print('label and counts',unique_labels, counts)

        # 找到包含点最多的类的类别号
        max_count_index = unique_labels[np.argmax(counts)]

        # 获取包含点最多的类的中心位置
        center_point = center_positions[max_count_index].astype(int)[::-1]

        # 可视化center_point

    else:
        '''到前置点的距离小于阈值的点的中心位置作为center'''
        '''# 计算每个点与前置点的距离
        distances = np.linalg.norm(points - pre_point, axis=1)
        
        # 找到距离小于等于阈值的点的索引
        valid_indices = np.where(distances <= thresh_dist)[0]
        
        # 提取符合条件的点的中心位置(平均/外接矩形中心)
        valid_points = points[valid_indices]
        
        # 如果valid_points为空，则在下一步计算时会出错，故需要特判
        if valid_points.size == 0:
            return 0, 0'''

        # 因为所有的点已经是有在domain之内这个限制了
        # 所以不需要再进行筛选。
        # 因此border的设定就很关键了，因为这限制了目标物的移动速度不能大于border*fps px/s
        # 因此border的设定应该与跳帧系数skip_n有关，与fps有关
        valid_points = points

        center_point = valid_points.mean(axis=0).astype(int)[::-1]

    '''center_point: (x, y)'''
    # print(center_point)
    border = 20
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
        '''注：在这里画完了，函数外面一样可以用'''
        cv.circle(frame,center_point,1,color=(0,0,255))
        # cv.rectangle(frame,(x_left, y_up),(x_right,y_down),(0,0,255),1)
        # if my_show(frame_show1, ratio=1, _time=0):
        #     return 'q',0
        pass
    
    return 'OK', center_point[::-1] # center_point: (y, x)

"""颜色识别的主函数"""
def main_color(cap,kind,root,OutWindow,progressBar,pm=1,skip_n=1):  # 颜色提取
    # main_color识别的border默认为20.
    # 是否超过border，即超出domain的范围，现在需要手动判断
    border = 20
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    ret, frame0 = cap.read()
    if not ret:
        showerror(message='读入错误')
        return 'error'
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), 
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))) 
    num = int(cap.get(7))
    Trc = Tractor()
    Trc.set("mutiple",pm)
    midval = np.zeros((3,))
    if kind == 'front':
        showinfo('','请提取前点颜色')
        midval_f = Trc.tract_color(frame0)
        if midval_f[0] == -1:
            return 'stop'
        if OutWindow and OutWindow.display:
            OutWindow.textboxprocess.delete('0.0','end')
            OutWindow.textboxprocess.insert('0.0','前点颜色(BGR)'+str(midval_f))
            OutWindow.textboxprocess.insert('0.0',"帧序号：[中心点坐标]\n")
            file = None
        else:
            file = open('out-color-1.txt','w')
            
        # showinfo(message='前点颜色(BGR)'+str(midval_f)+' ...请等待')
        midval = midval_f
    else:
        showinfo('','请提取后点颜色')
        midval_b = Trc.tract_color(frame0)
        if midval_b[0] == -1:
            return 'stop'
        if OutWindow and OutWindow.display:
            OutWindow.textboxprocess.delete('0.0','end')
            OutWindow.textboxprocess.insert('0.0','后点颜色(BGR)'+str(midval_b))
            OutWindow.textboxprocess.insert('0.0',"帧序号：[中心点坐标]\n")
            file = None
        else:
            file = open('out-color-2.txt','w')
            
        midval = midval_b
    
    cap.set(cv.CAP_PROP_POS_FRAMES, 0) # 重置为第一帧
    domain = (0,frame0.shape[0],0,frame0.shape[1]) # 上下左右
    domain = [int(x*pm) for x in domain]
    success = 1
    cnt = 0
    pre_state = -1 # 初始设为前一帧为black
    progressBar['maximum'] = num
    
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

            rtn, center = color_deal(frame[domain[0]:domain[1]+1,domain[2]:domain[3]+1],midval, 15, pre_state, np.linalg.norm(size) // 10, OutWindow)
            # thresh_dist = np.linalg.norm(size) // 10
            center_0 = (center[0]+domain[0], center[1]+domain[2])
            
            if rtn==0:
                pre_state = -1
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
                    '''中心点的圆点是在deal_color()中画的'''
                    cv.rectangle(frame_show, (domain[2],domain[0]), (domain[3],domain[1]), (0,0,255), 1)
                    domain_show = (domain[0]-border, domain[1]+border, domain[2]-border, domain[3]+border)
                    domain_show = restrict_to_boundary(domain_show, size[1],size[0])
                    if my_show(dcut(frame_show, domain_show),ratio=4, _time=100):
                        cv.destroyAllWindows()
                        break
                    printb(str(cnt)+': '+print_mid_point(center_0), OutWindow)
                else:
                    file.write(f'{cnt} {print_mid_point(center_0)}\n') # standard format

                pre_state = 1

            bar()
        # while
        
    if OutWindow and OutWindow.display:
        cv.destroyAllWindows() 
        pass
    else:
        file.close()
        showinfo(message='检测完成！')
    return 'ok'

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
    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 1, 1)

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
                # ret, track_window = cv.meanShift(dst, track_window, term_crit)
                ret, track_window = cv2.CamShift(dst, track_window, term_crit)

                center, size, angle = ret # 解包

                # 4.5 将追踪的位置绘制在视频上，并进行显示
                if OutWindow and OutWindow.display:
                    x, y, w, h = track_window
                    img2 = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
                    
                    rtn = my_show(img2,OutWindow.ratio, 60)
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
                # ret, track_window = cv.meanShift(dst, track_window, term_crit)
                ret, track_window = cv2.CamShift(dst, track_window, term_crit)

                center, size, angle = ret # 解包

                # 4.5 将追踪的位置绘制在视频上，并进行显示
                if OutWindow and OutWindow.display:
                    x, y, w, h = track_window
                    img2 = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
                    
                    if my_show(img2,OutWindow.ratio, 60):
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
    """get the magnify ratio in processing

    Args:
        cap (cv.CAP): the video
        root (tk.Tk): root window

    Returns:
        float, float, (int, int): body_length(px), real_length(cm), middle_point
    """
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    ret, frame0 = cap.read()
    Trc = Tractor()
    showinfo(message='请点击目标物前端')
    Trc.tractPoint(frame0, "please click the very front of the object, enter to confirm, q to quit, space to redo")
    point_f = Trc.gbPoint
    showinfo(message='请点击目标物后端')
    Trc.tractPoint(frame0, "please click the very back of the object, enter to confirm, q to quit, space to redo")
    point_b = Trc.gbPoint
    length = dist(point_f, point_b)
    Trc.inputbox(root=root,show_text='请输入目标物的实际长度，单位：厘米')
    body_length = eval(Trc.gbInput)
    # my_show(frame0, 50*body_length/length, midPoint(point_b,point_f))
    return body_length, length, midPoint(point_b,point_f)

'''之前写的旋转函数，现在用cv2的函数实现'''
def rotate(img, angle, center=None, scale=1.0):
    """
        @param:
            img: the img to be rotated
            angle: the angle to rotate the img (in degree)
            center: the center of the img
            scale: the scale of the img
    """
    # Determine the center of the img
    (h, w) = img.shape[:2]
    if center is None:
        center = (w/2, h/2)
    
    # Define the rotation matrix
    M = cv.getRotationMatrix2D(center, angle, scale)

    # Apply the rotation to the img
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
'''h,w = kernal.shape
x,y = pos
res = 0
for i in range(h):
    for j in range(w):
        res += frame[x+i][y+j]*kernal[i][j]
print(res)
return res'''

def max_conv2d(frame, domain, kernal, angle, display):
    """
    frame: 原图像
    domain: 卷积作用域
    kernal: 卷积核
    angle: 旋转角度
    display: 是否显示结果

    return value:
    max_pos_ori: (y, x)
    """

    # 转为弧度
    rad = angle * math.pi / 180

    # 旋转图像（逆时针旋转）
    rows, cols = frame.shape
    M = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated = cv.warpAffine(frame, M, (cols, rows))
    frame = rotated
    conv_res = cv.filter2D(frame, cv.CV_16S, kernal) # 整个图只卷积一次

    h = domain[1] - domain[0]
    w = domain[3] - domain[2]
    max_value = -1e7
    max_pos = (-1,-1)
    for y in range(domain[0],domain[1]+1):
        for x in range(domain[2],domain[3]+1):
            # 旋转对应点: 根据整个frame的中心旋转
            # anti-clockwise rotation(positive value)
            y1 = int((y - rows/2) * math.cos(rad) - (x - cols/2) * math.sin(rad) + rows/2)
            x1 = int((y - rows/2) * math.sin(rad) + (x - cols/2) * math.cos(rad) + cols/2)

            # 旋转之后有可能超出范围
            if not in_boundary((y1, x1), rows, cols):
                continue

            # 计算卷积
            res = conv_res[y1, x1]
            # frame[y1][x1] = numpy.sum(frame[y-domain[2]:y+domain[3]+1, x-domain[0]:x+domain[1]+1] * kernal)         
            # 更新最大值
            if res > max_value:
                max_value = res
                max_pos = (y, x)

    # 计算最大值对应的中心点
    max_pos_center = max_pos_ori = max_pos
    # h, w = kernal.shape
    # max_pos_center = (max_pos[0] + h//2, max_pos[1] + w//2)

    # if display:
    #     print('max_value: ' + str(max_value))
    #     print('max_pos: ' + str(max_pos))

    return max_pos, max_value # 回传中心点

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

"""特征提取的主函数"""
def feature(cap,kind='front',OutWindow=None,progressBar=None,root=None, skip_n=1, turn_start=1, turn_end=0) -> 0|1:
    """
    return: OK | stop
    """
    offset = 5
    cap.set(cv.CAP_PROP_POS_FRAMES, turn_start - 1)
    frame_num = cap.get(7) # 获取视频总帧数

    ret, frame0 = cap.read() # 读取第一帧
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))) # 获取视频尺寸
    # size的格式:(x, y)

    Idf = Identifier()
    showinfo(message='请拖动选择初始矩形框')
    rtn_ = Idf.select_window(frame0)
    
    (x,y,w,h), minis = rtn_ # minis未启用
    if x < 0:
        printb('用户取消', OutWindow)
        return 'stop'

    # 如果是展示模式，输出到窗口，否则打开文件句柄
    if OutWindow and OutWindow.display:
        printb("0 :", OutWindow)
    else:
        file = open('out-feature-1.txt','w') if kind == 'front' else open('out-feature-2.txt','w')

    domain = (y + h//2 - offset,y + h//2 + offset,x + w//2 - offset,x + w//2 + offset)
    max_angle = 0  # 初始的angle必须是0
    # max_angle_pos = max_conv2d(cv.cvtColor(frame0, cv.COLOR_BGR2GRAY),domain,Idf.K, 0, OutWindow and OutWindow.display)[0]
    # 初始化时预先找到一个中心点：False
    # domain = (max_angle_pos[0]-offset,max_angle_pos[0]+offset, max_angle_pos[1]-offset,max_angle_pos[1]+offset) # 更新domain

    # 初始化参数
    stdoutpb = Stdout_progressbar(frame_num, not(OutWindow and OutWindow.display))
    progressBar['maximum'] = frame_num
    success = 1
    # cnt = 0
    cnt = turn_start - 1

    cap.set(cv.CAP_PROP_POS_FRAMES, turn_start - 1)
    stdoutpb.reset(skip_n=skip_n)
    pre_angle = 0
    pre_state = 0  # init
    max_value_thresh = MAX_VALUE_THRESH
    # 或者为第一帧max_value的一半
    max_value = max_conv2d(cv.cvtColor(frame0, cv.COLOR_BGR2GRAY),domain,Idf.K, 0, OutWindow and OutWindow.display)[1]
    # max_value_thresh = max_value//2

    while success:
        success, frame = cap.read()
        if not success:
            stdoutpb.update(-1)
            break
        cnt += 1

        if turn_end > 0 and cnt > turn_end:
            break

        if skip_n > 1 and cnt % skip_n != 1:
            continue
        progressBar['value'] = cnt
        root.update()

        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        max_value = -1e6
        max_angle = pre_angle
        max_angle_pos = (-1, -1)

        angles = []
        if pre_state in [0, 1]:
            angles = list(range(pre_angle-5, pre_angle+5+1)) # 循环11次
        else:
            # angles = [-90, -60, -30, 0, 30, 60, 90] # 循环7次
            angles = [0] # 循环1次

        for angle in angles:
        # for angle in [pre_angle-5, pre_angle, pre_angle+5]: # 循环3次
            pos, value = max_conv2d(frame,domain,Idf.K, angle, OutWindow and OutWindow.display)
            if pos == (-1,-1): # abnormal status
                print('end(manually)')
                cv.destroyAllWindows()
                return 'stop'
            if value > max_value:
                max_value = value
                max_angle_pos = pos
                max_angle = angle

        if max_angle_pos == (-1, -1):
            cv.destroyAllWindows()
            raise Exception("error")

        if max_value < max_value_thresh:
            # black
            ob = OUTER_BORDER # int
            domain = (ob, size[1]-ob, ob, size[0]-ob)

            max_pos_center = (-1, -1)
            if OutWindow and OutWindow.display:
                if my_show(frame, ratio=1, _time=100):
                    return 'stop'
                printb('max_value: ' + str(max_value), OutWindow)
                printb('black', OutWindow)
                print('max_value:', max_value)
                print('black')
                print(f'{cnt} {"===" * 10} {cnt}')
            else:
                # 记录'black'
                file.write(f'{cnt} black\n')
                file.write(f'{cnt} relocate...\n')

            pre_angle = 0
            pre_state = 2 # black

            stdoutpb.update(cnt)
            continue


        domain = (max_angle_pos[0]-20,max_angle_pos[0]+20, max_angle_pos[1]-20,max_angle_pos[1]+20) # 更新domain

        # important:
        domain = restrict_to_boundary(domain, size[1], size[0])

        # max_pos_center = np.array(max_angle_pos) + np.array(Idf.K.shape)//2
        max_pos_center = max_angle_pos
        if OutWindow and OutWindow.display:
            frame_show = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
            frame_show = cv.circle(frame_show, max_pos_center[::-1], 3, (0, 0, 255))
            frame_show = dcut(frame_show, restrict_to_boundary((
                max_pos_center[0] - 40, max_pos_center[0] + 1 + 40, max_pos_center[1] - 40,
                max_pos_center[1] + 1 + 40), size[1], size[0]))

            if my_show(frame_show, ratio=4, _time=100):
                cv.destroyAllWindows()
                return 'stop'

            printb('max_value: ' + str(max_value), OutWindow)
            printb('now_angle: ' + str(max_angle), OutWindow)
            printb('now_pos: ' + str(max_pos_center), OutWindow)
            print('max_value:', max_value)
            print('now_pos:', str(max_pos_center))
            print('max_angle:', str(max_angle))
            print(f'{cnt} {"==="*10} {cnt}')
        else:
            # 记录中心点
            file.write(f'{cnt} {max_pos_center[0]},{max_pos_center[1]}\n') # standard format
            file.write(f'{cnt} angle {max_angle}\n') # 辅助角度变化
        pre_angle = max_angle
        pre_state = 1 # normal

        if OutWindow and OutWindow.display:
            printb(f'{cnt} {"==="*10} {cnt}', OutWindow)
        stdoutpb.update(cnt)

    # 结束
    showinfo(message='检测完成！')
    if OutWindow and OutWindow.display:
        cv.destroyAllWindows()
    else:
        file.close()
    return 'OK'

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
    showinfo(message='请拖动选择初始矩形框')
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
    
def tilt(edge, display, method='linear regression'):
    rect0, minmax_points = rect_cover(edge,55)
    
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
        if my_show(edge_show1, _time=100):
            cv2.destroyAllWindows()
            return (None,) * 3
    return rect, angle, len(points)

def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def to_hsv(img: Grayimg):
    # 创建一个与灰度图像相同大小的全零数组
    hsv_image = np.zeros(img.shape + (3,))

    # 将灰度图像复制到HSV图像的亮度通道
    hsv_image[:,:,0] = img
    hsv_image[:,:,1] = 1
    hsv_image[:,:,2] = 0.5
    
    return hsv_image

def edge(img: BGRimg, _operator='prewitt') -> Grayimg:
    """
        operator: sobel, prewitt, ... 
    """
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    if _operator == 'sobel':
        
        # 沿x方向的边缘检测
        img1 = cv.Sobel(img, cv.CV_64F, 1, 0)
        sobelx = cv.convertScaleAbs(img1)
        # 展示未进行取绝对值的图片(需要取绝对值)
        # cv_show('img1', img1)
        # cv_show('sobelx', sobelx)
        
        # 沿y方向的边缘检测
        img1 = cv.Sobel(img, cv.CV_64F, 0, 1)
        sobely = cv.convertScaleAbs(img1)
        # cv_show('sobely', sobely)

        sobelxy1 = cv.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        # cv_show('sobelxy1', sobelxy1)

        return sobelxy1
    
    elif _operator == 'roberts':

        roberts_x = cv2.filter2D(img, -1, np.array([[-1, 0], [0, 1]]))
        roberts_y = cv2.filter2D(img, -1, np.array([[0, -1], [1, 0]]))

        # 将水平和垂直边缘检测结果合并
        roberts_edges = cv.addWeighted(roberts_x, 0.5, roberts_y, 0.5, 0)

        # 显示原始图像和边缘检测结果
        # cv2.imshow('Original Img', img)
        # cv2.imshow('Roberts Edges', roberts_edges)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return roberts_edges
    
    elif _operator == 'prewitt':
        # 对图像进行Prewitt算子边缘检测
        prewitt_x = cv2.filter2D(img, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
        prewitt_y = cv2.filter2D(img, -1, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))

        # 将水平和垂直边缘检测结果合并
        prewitt_edges = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

        # 显示原始图像和边缘检测结果
        # cv2.imshow('Original img', img)
        # cv2.imshow('Prewitt Edges', prewitt_edges)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return prewitt_edges

    elif _operator == 'canny':
        v1=cv2.Canny(img,80,150)
        v2=cv2.Canny(img,50,100)
        
        res = np.hstack((v1,v2))
        # cv_show(res,'res')

        return res

    elif _operator == 'canny_gradient':
        # canny(): 边缘检测
        img1 = cv2.GaussianBlur(img,(3,3),0)
        canny = cv2.Canny(img1, 50, 150)

        # 形态学：边缘检测
        _,Thr_img = cv2.threshold(img,210,255,cv2.THRESH_BINARY)#设定红色通道阈值210（阈值影响梯度运算效果）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))         #定义矩形结构元素
        gradient = cv2.morphologyEx(Thr_img, cv2.MORPH_GRADIENT, kernel) #梯度

        # cv2.imshow("original_img", img) 
        # cv2.imshow("gradient", gradient) 
        # cv2.imshow('Canny', canny)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return gradient
        
"""轮廓方法识别主函数：Camshift"""
def contour_camshift(cap,background_img,root,OutWindow,progressBar,skip_n=1, turn_start=1,turn_end=0):
    # PROCESS HEAD
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    Trc = Tractor()
    ret, frame0 = cap.read()
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), 
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    frame_num = cap.get(7)
    if OutWindow and OutWindow.display:
        pass
    else:
        file_theta = open('out-camshift-theta.txt','w')
        file_center = open('out-camshift-center.txt','w')

    # PROCESS PREWORK
    showinfo(message='请拖动选择覆盖整个目标物的初始矩形框')
    r, h, c, w = Trc.select_rect(frame0)
    if r is None:
        return 'stop'

    x, y = c, r
    track_window = (x, y, w, h)
    domain = (r,r+h,c,c+w) # 上下左右
    
    roi = frame0[y:y + h, x:x + w]
    if background_img is None: # 不传入背景图，直接用camShift
        hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    else:
        edge1 = edge(roi)
        edge_background = edge(background_img)
        cha = cv.subtract(edge1, edge_background[y:y + h, x:x + w])
        hsv_roi = cv.cvtColor(cha, cv.COLOR_GRAY2BGR)
        hsv_roi = cv.cvtColor(hsv_roi, cv.COLOR_BGR2HSV)
        # hsv_roi = to_hsv(cha)

    if background_img is None:
        # 为了避免由于低光导致的错误值，使用 cv2.inRange() 函数丢弃低光值。
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    else:
        mask = cv2.inRange(cha, 100, 255)
    
    # if OutWindow and OutWindow.display:
        # if my_show(hsv_roi):
            # return 'stop'

    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        
    '''import matplotlib.pyplot as plt
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("% of Pixels")
    plt.plot(roi_hist)
    plt.xlim([0, 256])
    plt.show()'''

    '''没看出这个crit设置有什么区别'''
    # 设置终止标准，{n}次迭代或移动至少 1pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    # term_crit = (cv2.TERM_CRITERIA_EPS, 1, 1)

    # PROCESS INIT
    stdoutpb = Stdout_progressbar(frame_num, not(OutWindow and OutWindow.display))
    progressBar['maximum'] = frame_num
    success = 1
    cnt = turn_start - 1
    cap.set(cv.CAP_PROP_POS_FRAMES, turn_start - 1)
    stdoutpb.reset(skip_n=skip_n)

    # PROCESS MAINLOOP
    while (1):
        ret, frame = cap.read()

        if ret == True:
            # LOOP HEAD
            cnt += 1
            if skip_n > 1 and cnt % skip_n != 1:
                continue
            progressBar['value'] = cnt
            root.update()

            if background_img is None:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            else:
                cha = cv2.subtract(edge(frame), edge_background)
                hsv = cv2.cvtColor(cha, cv2.COLOR_GRAY2BGR)
                hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
                # hsv = to_hsv(cha)

            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # 应用camshift获取新位置
            # 返回一个旋转的矩形和框参数（用于在下一次迭代中作为搜索窗口传递）
            # 它首先应用均值变换。一旦meanshift收敛，它会更新窗口的大小，并且计算最佳拟合椭圆的方向。它再次应用具有新缩放搜索窗口和先前窗口位置的均值变换。该过程一直持续到满足所需的精度。
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)

            center, size, angle = ret # 解包

            if OutWindow and OutWindow.display:
                OutWindow.textboxprocess.insert('0.0', f'{center}  {round(angle,2)}\n')

                # 绘制在图像上
                pts = cv2.boxPoints(ret)
                pts = np.intp(pts)
                if background_img is None:
                    img2 = cv2.polylines(frame, [pts], True, 255, 2)
                else:
                    img2 = cv2.polylines(cha, [pts], True, 255, 2)
                cv2.imshow('img2', img2)
                k = cv2.waitKey(60) & 0xff
                if k == ord('q'): 
                    cv.destroyAllWindows()
                    return 'stop'
            else:
                file_center.write(f'{cnt} {str(center)[1:-1]}\n') # 去掉左右括号
                file_theta.write(f'{cnt} {round(angle,2)}\n')

            # LOOP TAIL
            stdoutpb.update(cnt)
        else:
            break
        
    # PROCESS TAIL
    if OutWindow and OutWindow.display:
        OutWindow.textboxprocess.insert('0.0','检测完成，展示模式不修改数据\n')
    else:
        file_center.close()
        file_theta.close()
        showinfo(message='检测完成!')
    cv2.destroyAllWindows()
    return 'OK'

"""轮廓方法识别主函数：线性回归"""
def contour_lr(cap,background_img,root,OutWindow,progressBar,skip_n=1, turn_start=1,turn_end=0):
    
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    Trc = Tractor()
    ret, frame0 = cap.read()
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), 
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    frame_num = cap.get(7)
    showinfo(message='请拖动选择初始矩形框')
    file_theta = open('out-contour-theta.txt','w')
    file_center = open('out-contour-center.txt','w')
    r, h, c, w = Trc.select_rect(frame0)
    if r is None:
        return 'stop'
    
    x, y = c, r
    track_window = (x, y, w, h)
    domain = (r,r+h,c,c+w) # 上下左右
    
    roi = frame0[y:y + h, x:x + w]
    edge1 = edge(roi)
    edge_background = edge(dcut(background_img,domain))
    cha = cv.subtract(edge1, edge_background)

    # cha: 保留了兼容性

    if OutWindow and OutWindow.display:
        if my_show(cha, _time=100):
            return 'stop'
    
    rect, angle, num = tilt(cha, OutWindow and OutWindow.display)
    if not rect:
        return 'stop'
    border = 40
    if num < 10:
        domain = (0, frame0.shape[0], 0, frame0.shape[1])
    else:
        domain = (rect[0]+domain[0]-border,rect[1]+domain[0]+border, rect[2]+domain[2]-border,rect[3]+domain[2]+border)
        domain = restrict_to_boundary(domain, size[1], size[0])

        if OutWindow and OutWindow.display:
            printb(f'angle: {angle}', OutWindow)
    
    # 初始化参数
    stdoutpb = Stdout_progressbar(frame_num, not(OutWindow and OutWindow.display))
    progressBar['maximum'] = frame_num
    success = 1
    cnt = turn_start - 1
    cap.set(cv.CAP_PROP_POS_FRAMES, cnt)
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

        cha = cv.subtract(edge(dcut(frame,domain)),edge(dcut(background_img,domain)))
        if OutWindow and OutWindow.display:
            OutWindow.textboxprocess.insert('0.0', f'{cnt}:\n')
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
                file_center.write(f'{cnt} 0 0\n')
                file_theta.write(f'{cnt} 0\n')
            continue
        domain = (rect[0]+domain[0]-border,rect[1]+domain[0]+border, rect[2]+domain[2]-border,rect[3]+domain[2]+border)
        domain = restrict_to_boundary(domain, size[1], size[0])
        if OutWindow and OutWindow.display:
            OutWindow.textboxprocess.insert('0.0', f'({print_mid_point(domain)})  {round(angle,2)}\n')
        else:
            file_center.write(f'{cnt} {print_mid_point(domain,sep=" ")}\n')
            file_theta.write(f'{cnt} {round(angle,2)}\n')
        stdoutpb.update(cnt)
        
    if OutWindow and OutWindow.display:
        OutWindow.textboxprocess.insert('0.0','检测完成，展示模式不修改数据\n')
    else:
        file_center.close()
        file_theta.close()
        showinfo(message='检测完成!')
    return 'OK'

def contour(cap,background_img,root,OutWindow,progressBar,skip_n=1, turn_start=1,turn_end=0, use_contour=False):
    """检测边缘，之后再计算角度"""
    if use_contour:
        return contour_lr(cap, background_img, root,OutWindow,progressBar,skip_n,turn_start,turn_end)

    # 使用camshift方法
    return contour_camshift(cap, background_img,root,OutWindow,progressBar,skip_n, turn_start,turn_end)

class FakeMs:
    def __init__(self) -> None:
        self.cnt = 0
    
    def update(self):
        self.cnt += 1

if pstatus == "debug":
    cap = cv2.VideoCapture(r"D:\GitHub\Cockroach-video-parse\src\前后颜色标记点.mp4")
    # cap = cv2.VideoCapture(r"C:\Users\LENOVO\Videos\10Hz，左，样本3 00_00_00-00_00_19.40_Trim.mp4")
    ret, frame0 = cap.read()
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    background = cv2.imread(r"D:\GitHub\Cockroach-video-parse\src\filename.png")
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
        # window.textboxprocess.insert("0.0", "111\n")
        # main_color(cap,'back',root=FakeMs(),OutWindow=window,progressBar=dict(),skip_n=10)
        # feature(cap,'back',OutWindow=window,progressBar=dict(),root=FakeMs(),skip_n=2, turn_start=1)
        # feature(cap,'back',progressBar=dict(),root=FakeMs(),skip_n=2, turn_start=1)
        contour_lr(cap,background,root=FakeMs(),OutWindow=None,progressBar=dict(),skip_n=1, turn_start=1)
        # contour(cap,None,root=FakeMs(),OutWindow=window,progressBar=dict(),skip_n=1, turn_start=1)

        # tier.mainloop()
