# Cockroach-video-parse

A python project to trace a cockroach in a video.

To pack an exe: pyinstaller -F [-n *filename*] main.py

To do List:

- [x] 输出转向角度angle, 
- [ ] 转向角速度omega_t, 摆动角速度omega_s,
- [ ] angle_interp...
- [x] 转弯半径radius
- [x] 转向半径单位：米
- [x] main window size(change)
- [x] a drag rectangle when selecting a rect
- [x] result-file-output: 'black'
- [x] format cv to cv2 for all the files
- [x] put ResWindow on the front
- [x] show angle first frame be 0?
- [x] resolve the error occurred when loading a video
- [ ] main window size not stable?
- [ ] sequence of plots
- [ ] tract point show output to out_window
- [ ] add cut_edge when tracting points
- [ ] check if the multiple process is OK
- [ ] complete feature_detection
- [x] rewrite prompt(select_window)
- [ ] solve the filename problem
- [x] change to the dynamic threshold for radius
- [ ] choice: timestamp should be the time when detecting goes or data_dealing(result_viewing)?
- [ ] add detecting_mark_str record
- [x] refresh property when loading a new video
- [ ] regular imports?
- [x] add "release" and "debug" mode to all files
- [x] os.makedir
- [x] regular output file path
- [x] plt.figure(x) - regular figure name
- [ ] delete bokeh-format codes
- [x] put pstatus in front of each file
- [x] resolve the problem of "can't go plt.show()"
- [ ] function: load_last_result
- [x] some problems about px_to_cm result dealing
- [ ] regular plot: x_label and y_label for each plot

  (xlabel, ylabel, figname, title)

- [ ] some problems about display_window

- [ ] put dispWindow to the top when detecting
- [x] add prompt: 提取过程中止
- [ ] WindowsLift
- [ ] a pause button ({space})
- [ ] outWindow open and close
- [ ] contour-detection my_show close error
- [x] central-control mode ["release","debug"]
- [ ] Dynamic window size
- [ ] update display_window



My Log:

5.25 afternoon update radius result output(file:deal_with_data.py) and not push yet and delete file...

5.29 I wonder why the scale of window changed (as former 1200x700 now 1600x850 to the same size)