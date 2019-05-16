import cv2
import sys
import numpy as np

def roiDetector(eye, frame):
    # ret, eye = video_capture.read()#boolean, eye
    # h, w = eye.shape
    roi = eye.copy()
    rows, cols = roi.shape
    # cv2.imshow('v', eye)
    cv2.imshow('roi', roi)
    # gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.equalizeHist(eye)
    gray_roi = cv2.GaussianBlur(gray_roi, (7 ,7), 0)
    # gray_roi = cv2.equalizeHist(gray_roi)
    _, threshold = cv2.threshold(gray_roi, 5, 255, cv2.THRESH_BINARY_INV)#若無equalize:40
    kernal = np.ones((rows, cols),np.uint8)
    cv2.imshow('before', threshold)
    ersThre = cv2.erode(threshold, kernal)
    # threshold = cv2.dilate(ersThre, kernal)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for cnt in contours: cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3) #draw contours
    ###-----------------from net
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)###
    for cnt in contours:
        (xx, yy, ww, hh) = cv2.boundingRect(cnt)
       #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
        xx = xx+x
        yy = yy+y
        cv2.rectangle(frame, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)
        # cv2.line(frame, (xx + int(ww/2), 0), (xx + int(ww/2), rows), (0, 255, 0), 2)
        # cv2.line(frame, (0, yy + int(hh/2)), (cols, yy + int(hh/2)), (0, 255, 0), 2)
        break
    
    ###-------------
    # for cnt in contours:
        # cv2.drawContours(roi, [cnt], -1, (0,0,255), 3)
    # cv2.imshow('roi', roi)
    cv2.imshow('gray_roi', gray_roi)
    cv2.imshow('thre', threshold)
    return frame




cascPath = "haarcascade_frontalface_default.xml"#路徑
cascPath1 = "haar_eye.xml"
faceCascade = cv2.CascadeClassifier(cascPath)#用這個分類去找出來有沒有符合
eyeCascade = cv2.CascadeClassifier(cascPath1)

video_capture = cv2.VideoCapture(1)
# video_capture = cv2.VideoCapture('skin_example.mov')
frame = video_capture.read()[1]#boolean, frame
height, width = frame.shape[:2]
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()#boolean, frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ###---------optional
    # gray = cv2.equalizeHist(gray)#好像並沒有效果特別好
    #-------------
    # h, w = gray.shape
    # linstrimg = gray.copy()
    # p1 = (150, 0)
    # p2 = (210, 190)
    # for i in range(h):
    #     for j in range(w):
    #         # linstrimg[i, j] = math.log(gray[i, j]+1)*255 / math.log(255) #log()
    #         if gray[i, j] < p1[0]:
    #             linstrimg[i, j] = gray[i, j] * p1    [1] // p1[0]
    #         elif gray[i, j] > p2[0]:
    #                 linstrimg[i, j] = p2[1] + (gray[i, j] - p2[0]) * (255 - p2[1]) // (255 - p2[0])
    #         else:
    #                 linstrimg[i, j] = p1[1] + (gray[i, j] - p1[0]) * (p2[1] - p1[1]) // (p2[0] - p1[0])

    #-------------
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,#每次掃過增加的面積倍數
        minNeighbors=5,#至少鄰近有符合特徵的數量
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE###修正過
    )

    img = np.zeros((height,width,1), np.uint8)
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        h = int(0.7*h)#發現他只吃整數
        img = cv2.rectangle(img, (x-10, y-10), (x+w+10, y+h), 255, -1)#框框(哪張圖,左上角位置,左下角位置,顏色,寬度)
        # img = cv2.rectangle(img, (x, y), (x+w, y+h), 255, -1)#原本的
        # cv2.imshow('rec', img)
    if faces != (): mask_faces = cv2.bitwise_and(gray, gray, mask = img)#位元運算
    else: mask_faces = gray
    # cv2.imshow('mask_faces', mask_faces)
    #eyes detect
    eyes = eyeCascade.detectMultiScale(
        mask_faces,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    img = np.zeros((height,width,1), np.uint8)
    # mask_roi = []
    # i = 0
    for (x, y, w, h) in eyes:
        mask_roi = gray[y:(y+h), x:(x+w)]
        # i += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)#原式
        # mask_roi = cv2.bitwise_and(gray, gray, mask = img)
    # cv2.imshow('maskroi', mask_roi)
        # roi[i] = frame[x:(x+w), y:(y+h)]
        # roiDetector(mask_roi)
        frame = roiDetector(mask_roi, frame)
    # cv2.imshow('mask', mask_roi)
    # cv2.imshow('img', img)
    ###--------------roi:
    # roi = frame[220: 500, 450: 800]

    # rows, cols, _ = roi.shape
    # cv2.imshow('roi', roi)
    # gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # gray_roi = cv2.equalizeHist(gray_roi)
    # gray_roi = cv2.GaussianBlur(gray_roi, (7 ,7), 0)##網路上學到的
    # # gray_roi = cv2.equalizeHist(gray_roi)
    # _, threshold = cv2.threshold(gray_roi, 3, 255, cv2.THRESH_BINARY_INV)#若無equalize:40
    # contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # ###-----------------from net
    # contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)###
    # for cnt in contours:
    #     (x, y, w, h) = cv2.boundingRect(cnt)
    #    #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
    #     cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #     cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
    #     cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
    #     break
    # Display the resulting frame
    cv2.imshow('Video', frame)
    cv2.imshow('gray', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


#下一步:
# 先實驗在roi檔案中能否畫線在原始照片上
# 在roiDetector引入原本照片，讓畫線時直接畫在原照片上(先用img，黑色的那個代替)，還要引入眼睛在原始照片的座標


