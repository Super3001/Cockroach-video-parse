import cv2 # 定义鼠标回调函数

x_start, y_start, x_end, y_end, drawing = (0,)*5

def mouse_callback(event, x, y, flags, param):    
    global x_start, y_start, x_end, y_end, drawing 
    img2 = img.copy()   
    print('event',event)
    print('flags',flags)
    if event == 1: # cv2.EVENT_LBUTTONDOWN = 1
        # drawing = True        
        x_start, y_start = x, y    
    elif event == 0 and flags == 1: # cv2.EVENT_MOUSEMOVE = 0                   
        x_end, y_end = x, y    
        cv2.rectangle(img2, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.imshow('select',img2)
    elif event == 4: # cv2.EVENT_LBUTTONUP = 4              
        x_end, y_end = x, y
        print(f'gbRect: {(x_start, y_start, x_end, y_end)}')
        inpt = input('check?')
        if(inpt):
            cv2.destroyWindow('select')
            return (x_start, y_start), (x_end, y_end)
        else:
            cv2.destroyWindow('select')
    """debug interation"""
    if event == cv2.EVENT_MOUSEWHEEL:
        print(x,y)
        print(flags)
        print(f'gbRect: {(x_start, y_start, x_end, y_end)}')     

"""
def mouse_callback(event, x, y, flags, param):    
    global x_start, y_start, x_end, y_end, drawing 
    img2 = img.copy()   
    print('event',event)
    print('flags',flags)
    if event == 1: # cv2.EVENT_LBUTTONDOWN = 1
        drawing = True        
        x_start, y_start = x, y    
    elif event == 0 and flags == 1: # cv2.EVENT_MOUSEMOVE = 0        
        if drawing:            
            x_end, y_end = x, y    
            cv2.rectangle(img2, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.imshow('select',img2)
    elif event == cv2.EVENT_RBUTTONDOWN:        
        drawing = False        
        x_end, y_end = x, y
        cv2.destroyWindow('select')
    if event == cv2.EVENT_MOUSEWHEEL:
        print(x,y)
        print(flags)
        print(f'gbRect: {(x_start, y_start, x_end, y_end)}')   
"""   
    
# 创建窗口并显示图片
img = cv2.imread('src//background.png')
cv2.namedWindow('image')
cv2.imshow('image', img) 
# 注册鼠标回调函数
cv2.setMouseCallback('image', mouse_callback)# 主循环
drawing = False
while True:    
    cv2.imshow('image', img)    
    key = cv2.waitKey(1) & 0xFF
    if key < 255:
        print(key)
    if key == ord('q'):          
        break
    elif key == 13: # {ENTER}          
        print((x_start, y_start), (x_end, y_end))
cv2.destroyAllWindows()