import cv2
import time
import os

f = open('mouse_track.txt','w')

def write(s:str):
    f.write(s+'\n')

# 定义鼠标事件回调函数
def mouse_callback(event, x, y, flags, param):
    global mouse_pos, mouse_track

    # if event == cv2.EVENT_LBUTTONDOWN:
    # 计算鼠标相对位置
    relative_pos = [x - mouse_pos[0], y - mouse_pos[1]]

    # 不需更新鼠标位置
    # mouse_pos = [x, y]

    # 记录鼠标移动轨迹
    mouse_track.append((x, y))

    # 在图像上显示鼠标相对位置
    # cv2.putText(img, "Relative Position: ({}, {})".format(relative_pos[0], relative_pos[1]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    print(relative_pos)
    write(f'{int((time.time() - st)*1000)} {relative_pos}')

    # 在图像上显示鼠标移动轨迹
    for i in range(len(mouse_track) - 1):
        cv2.line(img, mouse_track[i], mouse_track[i + 1], (0, 0, 255), 2)
    # time.sleep(0.01)
    
# if not os.path.exists(r".\src\white.png"):
#     if not os.path.exists(r".\src"):
#         os.mkdir('.\src')
    
#     import requests
#     from PIL import Image
#     from io import BytesIO
    
#     response = requests.get("https://gitee.com/syf2687/images/blob/master/white.png")
#     image = Image.open(BytesIO(response.content))
#     image.save(r'.\src\white.png')


# 创建窗口和图像
cv2.namedWindow("Mouse Tracker")
img = cv2.imread(r".\src\white.png")
# cv2.imshow()

# 设置鼠标初始位置
mouse_pos = [0, 0]

# 设置鼠标移动轨迹
mouse_track = []

# 注册鼠标事件回调函数
cv2.setMouseCallback("Mouse Tracker", mouse_callback)

write(time.strftime("%D-%H%M%S", time.localtime()))
st = time.time()

# 显示图像
while True:
    cv2.imshow("Mouse Tracker", img)
    cv2.moveWindow("Mouse Tracker", 0, 0)

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

# 释放窗口和图像
cv2.destroyAllWindows()

f.close()