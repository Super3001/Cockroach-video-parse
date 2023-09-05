# Cockroach-video-parse

A python project to trace a cockroach in a video.

To pack an exe: pyinstaller -F [-n *filename*] main.py

## Note：
1. 如果蟑螂出镜了，或者被手挡住了，则那一段的检测结果可能会失效。请务必注意这一点。
2. 标志点和环境没有明显颜色差异的，不要用color
3. 使用轮廓识别需要预先处理好背景图，可以利用“展示第一帧”功能导出第一帧图像，之后再进行抠图处理
4. 使用轮廓识别不能输出摆动角速度
5. 四种识别方法的比较

| method      | 速度 | 精度 | 重新定位 |
| :--:        | :--: | :--: | :--:    |
| meanshift   | 快   | 较高 | ok | 
| color       | 较快 | 较高 | ok |
| contour     | 较慢 | 一般 | ok |
| contour+cnn | 较慢 | 较高 | ok |
| feature     | 较快 | 高   | 困难 |

## 功能实现：

| method      | basic | mutiple | skip read | 
| :--:        | :--:  | :--:    | :--:      |
| meanshift   | OK | OK | OK |
| color       | OK | OK | OK |
| contour     | OK | -  | OK |
| contour+CNN | -  | -  | -  |
| feature     | OK | -  | OK |

| data object | output to file | output to plt | change unit |
| :--:        | :--:           | :--:          | :--:        |
| path        | ok | ok | ok |
| radius      | ok | ok | ok |
| angle       | ok | ok | -  |
| omega       | ok | ok | -  |


## 代码特征：
1. 传文件传文件句柄
2. 中间文件和最终输出文件统一到一个格式（待完成）
3. 统一用i或者idx（或者pf）表示下标，f或者frame表示帧数


4. 记录数据的格式标准为int或者float一位小数（根据精度）. '{cnt} {value1,value2,...}'
5. 处理数据的格式标准为... np.ndarray

## To do List:

- [x] 输出转向角度angle, 

- [x] 转向角速度omega_t, 摆动角速度omega_s,

- [x] angle_interp...

- [x] 转弯半径radius

- [x] 转向半径单位：米

- [x] main window size(change)

- [x] a drag rectangle when selecting a rect

- [x] result-file-output: 'black'

- [x] format cv to cv2 for all the files

- [x] put ResWindow on the front

- [x] show angle first frame be 0?

- [x] resolve the error occurred when loading a video

- [x] main window size not stable?

- [x] sequence of plots

- [ ] tract point show output to out_window

- [x] add cut_edge when tracting points

- [x] check if the multiple process is OK

- [x] complete feature_detection

- [x] rewrite prompt(select_window)

- [x] solve the filename problem

- [x] change to the dynamic threshold for radius

- [ ] choice: timestamp should be the time when detecting goes or data_dealing(result_viewing):

    目前是跟着ResWindow走的

- [x] add detecting_mark_str record

- [x] refresh property when loading a new video

- [x] regular imports:

    except for tkinterUI.py

- [x] add "release" and "debug" mode to all files

- [x] os.makedir

- [x] regular output file path

- [x] plt.figure(x) - regular figure name

- delete bokeh-format codes(/)

- [x] put pstatus in front of each file

- [x] resolve the problem of "can't go plt.show()"

- [ ] function: load_last_result

- [x] some problems about px_to_cm result dealing

- [x] regular plot: x_label and y_label for each plot

  (xlabel, ylabel, figname, title)

- [ ] some problems about display_window

- put dispWindow to the top when detecting（/）

- [x] add prompt: 提取过程中止

-  WindowsLift（/）

- [x] a pause button ({space})

- [x] outWindow open and close

- [x] contour-detection my_show close error

- [x] central-control mode ["release","debug"]

- [x] update display_window

- write about *minis*, balance accuracy with time(/)

- [x] 读取过程缩放-功能实现（处理过程缩放）

- [x] 处理过程中写文件，要把帧数写上，以适应skip read功能

- [x] 把deal_with_data拆分为parse_data和deal_data两个文件

- [x] 是否需要parse data过程中都带着frame number -> 所有的数据都进行有效化（

      X1, Y1, X2, Y2是未过滤的，通过self.frames进行过滤
    
      X_mid, Y_mid, K, D, Theta都是过滤之后的
    
      这样统一了有效数据的帧数，可以用indice和mask写法，优先使用mask）

- [x] 数据np.array化

- 放大的选点/选颜色(/)

- [x] 把QtUI去掉

- [x] color 和 feature的提取过程展示

- [x] show first frame & save

7.14 consult with Yuli:
- [x] 调整kernal生成方式
- [x] 标志点特征旋转
- [ ] 窗口大小实现响应式布局
- [x] 角度结果跳变

9.1
- [x] 加入辅助角度

9.3
- [ ] update 轮廓识别(?)


My Log:

5.25 afternoon update radius result output(file:deal_with_data.py) and not push yet and delete file...

5.29 I wonder why the scale of window changed (as former 1200x700 now 1600x850 to the same size)

6.17 write skip_read format like this: 

    ```
    if skip_n > 1 and cnt % skip_n != 1:
    
        continue
    ```

  7.10 write stdout_progessbar like this:

    ```
    初始化：
    stdoutpb = Stdout_progressbar(num, not(OutWindow and OutWindow.display))
    cnt = 0
    stdoutpb.reset(skip_n)
    
    每次迭代：
      cnt += 1
      ...(process)
      stdoutpb.update(cnt)
    
    结束迭代：
      if ...(not success):
        stdoutpb.update(-1)
        break
    ```

7.11 pm不一定是整数，但可能会有误差 
