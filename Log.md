# Log (Cockroadch video parse)

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



8.19.2023 

version 0.7.0


8.30.2023

version 0.7.1

- change feature(not examine yet)
- change plot

8.31.2023

- change default light show time = 0.1

- add cv.destroyAllWindows() many places

- change `color` using kmeans/mean in dist_thresh

格式要保持一致，不然可能会出错

9.1.2023

- color中的聚类那部分暂时认为是没有错的（待验证）

- 取消了一些frame_show

- 四种方法的主函数都加上`kind`参数

- 加入turn_start和turn_end

9.3.2023

- 修改了画图逻辑（主要是showCurve）和代码

9.6.2023

- contour(lr)写得不规范

- 隐藏方法：camshift

9.9.2023

- tract-point应该适应一下几种情况：
  1. 直接关闭窗口
  2. 按q关闭窗口
  3. 选择之后点击确定
  4. 选择之后点击取消
  5. 第一次选择之后按其他键
  6. 第一次选择之后直接关闭roi

9.16.2023

1. 删去`倍数缩放处理`

