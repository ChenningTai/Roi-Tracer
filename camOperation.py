import cv2
import sys
import math
from matplotlib import pyplot as plt
import numpy as np

video_capture = cv2.VideoCapture(1)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()#boolean, frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # print(hsv)
    h, s, v = cv2.split(hsv)
    hi, w = h.shape


    kernal = np.ones((5, 5),np.uint8)
    clsS = cv2.morphologyEx(s, cv2.MORPH_CLOSE, kernal)


    part = np.ones((hi,w,1), np.uint8)*200  #可直接乘，乘法函數應該有重新定義了
    # part = s.fill(200)  #不知為何出錯
    # print(part)
    hzz = cv2.merge((h, part, part))
    # print(part)
    hBGR = cv2.cvtColor(hzz, cv2.COLOR_HSV2BGR)
    # h1 = cv2.equalizeHist(h)
    
    ss = np.hstack((s, clsS))

    cv2.imshow('video', frame)
    cv2.imshow('hBGR', hBGR)
    cv2.imshow('s', ss)
    cv2.imshow('v', v)
    cv2.imshow('Video', hsv)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


# print map(int,listA)
# print [multiple2(x) for x in list1]