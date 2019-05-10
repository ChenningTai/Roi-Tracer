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
    rows, cols, _ = roi.shape
    # cv2.imshow('v', frame)
    cv2.imshow('roi', roi)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.equalizeHist(gray_roi)
    gray_roi = cv2.GaussianBlur(gray_roi, (7 ,7), 0)##網路上學到的
    # gray_roi = cv2.equalizeHist(gray_roi)
    cv2.imshow('gray_roi', gray_roi)
    _, threshold = cv2.threshold(gray_roi, 3, 255, cv2.THRESH_BINARY_INV)#若無equalize:40
    cv2.imshow('thre', threshold)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ###-----------------from net
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)###
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
       #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
        break
    ###-------------
    # for cnt in contours:
        # cv2.drawContours(roi, [cnt], -1, (0,0,255), 3)
    cv2.imshow('roi', roi)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


# # # print map(int,listA)
# # # print [multiple2(x) for x in list1]

# import cv2
# import numpy as np

# cap = cv2.VideoCapture("eyeVideo.mp4")

# while True:
#     ret, frame = cap.read()
#     if ret is False:
#         break

#     # roi = frame[269: 795, 537: 1416]
#     # rows, cols, _ = roi.shape
#     # gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     # gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

#     roi = frame[220: 500, 450: 800]
#     rows, cols, _ = roi.shape
#     # cv2.imshow('v', frame)
#     cv2.imshow('roi', roi)
#     gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     gray_roi = cv2.equalizeHist(gray_roi)
#     gray_roi = cv2.GaussianBlur(gray_roi, (7 ,7), 0)##網路上學到的

#     _, threshold = cv2.threshold(gray_roi, 3, 255, cv2.THRESH_BINARY_INV)
#     contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

#     for cnt in contours:
#         (x, y, w, h) = cv2.boundingRect(cnt)

#         #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
#         cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
#         cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
#         break

#     cv2.imshow("Threshold", threshold)
#     cv2.imshow("gray roi", gray_roi)
#     cv2.imshow("Roi", roi)
#     key = cv2.waitKey(30)
#     if key == 27:
#         break

# cv2.destroyAllWindows()