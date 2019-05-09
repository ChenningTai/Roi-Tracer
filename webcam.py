import cv2
import sys
import numpy as np
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
    eyes = eyeCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    img = np.zeros((height,width,1), np.uint8)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:

        img = cv2.rectangle(img, (x-10, y-10), (x+w+15, y+h+15), 255, -1)#框框(哪張圖,左上角位置,左下角位置,顏色,寬度)
        # img = cv2.rectangle(img, (x, y), (x+w, y+h), 255, -1)
        cv2.imshow('rec', img)
        # print(type(img))
        frame = cv2.bitwise_and(frame, frame, mask = img)#位元運算
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
