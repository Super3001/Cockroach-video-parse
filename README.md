# Cockroach-video-parse

A python project to trace a cockroach in a video.

To pack an exe: pyinstaller -F [-n *filename*] main.py

To do List:

- [x] 输出转向角度angle, 
- [ ] 转向角速度omega_t, 摆动角速度omega_s, 
- [ ] 转弯半径radius
- [x] 转向半径单位：米
- [x] main window size(change)
- [ ] a drag rectangle when selecting a rect
- [ ] sequence of plots
- [x] result-file-output: 'black'
- [ ] main window size not stable?