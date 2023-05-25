# Cockroach-video-parse

A python project to trace a cockroach in a video.

To pack an exe: pyinstaller -F [-n *filename*] main.py

To do List:

- [x] 输出转向角度angle, 
- [ ] 转向角速度omega_t, 摆动角速度omega_s, 
- [x] 转弯半径radius
- [x] 转向半径单位：米
- [x] main window size(change)
- [x] a drag rectangle when selecting a rect
- [x] result-file-output: 'black'
- [x] format cv to cv2 for all the files
- [ ] main window size not stable?
- [ ] sequence of plots
- [ ] tract point show output to out_eindow
- [ ] add cut_edge when tracting points
- [ ] check if the multiple process is OK
- [ ] complete feature_detection
- [x] put ResWindow on the front
- [x] show angle first frame be 0?
- [ ] rewrite prompt(select_window)
- [ ] solve the fliename problem
- [ ] change to the dynamic threshold for radius
- [x] resolve the error occurred when loading a video
- [ ] choice: timestamp should be the time when detecting goes or data_dealing(result_viewing)?
- [ ] add detecting_mark_str record
- [ ] refresh property when loading a new video


My Log:
5.25 afternoon update radius result output(file:deal_with_data.py) and not push yet and delete file...