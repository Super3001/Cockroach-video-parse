# CVP program debug

## 现存的一些问题：

滑动窗口普遍存在抖动的问题

特征标记点 250-350f

## 说明书的补充内容：

1. curvature的逐帧记录未出现的帧与前一帧的曲率相同（用的是过三点的圆的方法）

2. 转换单位/取消转换单位生效需要重新点击“查看结果”

3. 转换单位也可以依据标尺来做

4. (1)轮廓检测尽量用于浅色纯色背景，可以提前拍摄背景图，背景固定不动，(2)目标物有较为明显的（两侧对称的）长轴 **或** (?)

## debug记录

![image-20230831100922669](D:\GitHub\Cockroach-video-parse\src\md_img\image-20230831100922669.png)

2. thresh

3. output window

4. color的异常处理

5. destroywindows前的判断

9.1.2023

1. 因为卷积的方法换成了cv.filter2D, conv_res也变成了中心点。要以中心点位置来操作

2. feature做的value如果小于某个值，判定为black（异常处理）

3. 不用想初始值的事情了

![domain not restrict to boundary](image.png)

4. 按q退出

9.9.2023

1. 文件句柄的打开与否与是否打开展示有关！

2. & 0xFF error[x]

3. lift报错问题

4. destroyWindow报错问题

5. OutWindow已关闭报错问题(feature)

9.16.2023

1. cv显示中文的问题

2. 保存数据格式的问题

10.1.2023

1.一般识别（前后点）改回meanshift

10.8.2023

1. "刺激" 改为 "控制信号"

![image-20231008165744480](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20231008165744480.png)

![image-20231008180433238](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20231008180433238.png)

10.9.2023

OutputWindow的滚动条问题

展示输出到控制台？打开控制台progressbar？