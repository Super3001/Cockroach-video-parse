# CVP program debug



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

