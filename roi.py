import cv2
import sys
import math
from matplotlib import pyplot as plt
import numpy as np

video_capture = cv2.VideoCapture('eyeVideo.mp4')
# video_capture = cv2.VideoCapture(1)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()#boolean, frame
    # h, w = frame.shape
    roi = frame[220: 500, 450: 800]
    # cv2.imshow('v', frame)
    cv2.imshow('roi', roi)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.equalizeHist(gray_roi)
    gray_roi = cv2.GaussianBlur(gray_roi, (7 ,7), 0)##網路上學到的
    # gray_roi = cv2.equalizeHist(gray_roi)
    cv2.imshow('gray_roi', gray_roi)
    _, threshold = cv2.threshold(gray_roi, 10, 255, cv2.THRESH_BINARY_INV)#若無equalize:40
    cv2.imshow('thre', threshold)
    _, contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(roi, [cnt], -1, (0,0,255), 3)
    cv2.imshow('roi', roi)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


# print map(int,listA)
# print [multiple2(x) for x in list1]