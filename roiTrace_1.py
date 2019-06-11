import cv2
import numpy as np
import dlib
from math import hypot
import time

def roiDetector(gray_Eye, frame, x, y):#眼睛中畫出瞳孔傳回座標
    h, _ = gray_Eye.shape
    roi = gray_Eye.copy()
    roi0 = cv2.equalizeHist(roi)
    roi1 = cv2.GaussianBlur(roi0, (7 ,7), 0)
    roi = cv2.equalizeHist(roi1)
    _, threshold = cv2.threshold(roi, 40, 255, cv2.THRESH_BINARY_INV)
    
    kernal = np.ones((2, 2),np.uint8)
    threshold = cv2.dilate(threshold, kernal)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ##-----------------draw rec on Contours
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)###
    for cnt in contours:
        (xx, yy, ww, hh) = cv2.boundingRect(cnt)
        centerPosi_x = xx + ww/2
        # centerPosi_y = yy + hh/2
        centerPosi_y = h
        xx = xx+x
        yy = yy+y-5
        cv2.rectangle(frame, (xx, yy), (xx + ww, yy + hh), (0, 0, 255), 2)
        break
# # method 2 to draw contour center-------------------------
#     if len(contours) > 1:
#         maxArea = 0
#         MAindex = 0
#         distanceX = []
#         currentIndex = 0
#         for cnt in contours:
#             area = cv2.contourArea(cnt)
#             center = cv2.moments(cnt)
#             print(center)
#             cx = int(center['m10']/center['m00'])
#             cy = int(center['m01']/center['m00'])
#             print(cx, cy)
#             distanceX.append(cx)
#             if area > maxArea:
#                 maxArea = area
#                 MAindex = currentIndex
#             currentIndex = currentIndex + 1
#         # del contours[MAindex]
#         # del distanceX[MAindex]
# #------------------------------------

    return frame, np.hstack([roi0, roi1, roi, threshold]), centerPosi_x, centerPosi_y

def get_eye(eye_points, facial_landmarks,frame):#找出眼睛位置
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
#-------------------
    y = center_top[1]
    x = left_point[0]
    eye = frame[(y-5):center_bottom[1],x:right_point[0]]
    gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    frame, analysis, centerPosi_x, centerPosi_y = roiDetector(gray_eye, frame, x, y)
#計算瞳孔位置比例:
    widRatio = centerPosi_x / (right_point[0]-x)
    heiRatio = centerPosi_y
#-------------------
    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
    return np.hstack([gray_eye, analysis]), [widRatio, heiRatio]


def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def Pointer(ratio):#傳入座標[[右眼寬,右眼高], [左眼寬,左眼高]] #瞳孔位置投影螢幕位置
#先右後左
    Left = [upLeft[0],upLeft[2]]
    Right = [downRight[0], downRight[2]]
    Up = [upLeft[1], upLeft[3]]
    Down = [downRight[1], downRight[3]]

    x = [-1,-1]
    y = [-1,-1]
    #推關係式(未來以xy映射線性空間)
    for i in range(2):
        x[i] = (ratio[i][0]-Left[i])/(Right[i]-Left[i])
        y[i] = (ratio[i][1]-Up[i])/(Down[i]-Up[i])
    return int(sum(x)*960), int(sum(y)*540)#兩眼資料取平均(這樣是最有效的嗎??) #將原公式*1920/2 改寫成*960，*540抑是

def Correction():#執行前校正 參考點
    posiList = [[0.5]*64,[0.5]*64,[0.5]*64,[0.5]*64]#若使用[[0.5]*64]*4則會錯將外層[]中四個元素指向同一個參考位置
    mean = []
    for i in range(64):
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            centerRatio = [None, None]
            analysisR, centerRatio[0] = get_eye([36, 37, 38, 39, 40, 41], landmarks, frame)
            analysisL, centerRatio[1] = get_eye([42, 43, 44, 45, 46, 47], landmarks, frame)
            for n in range(2):
                for m in range(2):
                    k = n*2+m
                    posiList[k][i] = centerRatio[n][m]
#除去極端(誤差)值後算平均:
    for each in posiList:
        each.sort()
        mean.append(sum(each[17:48])/32)
    return mean


#Main():
# cap = cv2.VideoCapture('news.mp4')
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#Correction:
print('Please stare at the up left edge...')
time.sleep(1)
upLeft = Correction()
print('please stare at the down right edge...')
time.sleep(1)
downRight = Correction()
#prepare:
ini = 0.5
eightFotoX = [ini]*8#for Purify()
eightFotoY = [ini]*8#for Purify()
i = 0
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    blk = np.zeros((1080,1920), np.uint8)
    for face in faces:
        landmarks = predictor(gray, face)
        centerRatio = [None, None]
        analysisR, centerRatio[0] = get_eye([36, 37, 38, 39, 40, 41], landmarks, frame)#這邊讀取傳回直應該將左右分法改成xy分法
        analysisL, centerRatio[1] = get_eye([42, 43, 44, 45, 46, 47], landmarks, frame)
        sightPoint_X, sightPoint_Y = Pointer(centerRatio)
#Purification():
        j = i%8
        eightFotoX[j] = sightPoint_X#兩眼資料取平均(這樣是最有效的嗎??) 
        eightFotoX_u = int(sum(eightFotoX)/8)
        eightFotoY[j] = sightPoint_Y #兩眼資料取平均(這樣是最有效的嗎??)
        eightFotoY_u = int(sum(eightFotoY)/8)
        i += 1

#analysis
        h1, w1 = analysisR.shape
        h2, w2 = analysisL.shape
        blk[0:h1, 0:w1] = analysisR
        blk[50:(h2+50), 0:w2] = analysisL
        cv2.circle(blk, (eightFotoX_u,eightFotoY_u), 10, (255,0,0), 2) #, lineType=None, shift=None)
        # cv2.circle(blk, (sightPoint_X,sightPoint_Y), 10, (255,0,0), -1) #, lineType=None, shift=None) #原始無purify點
        cv2.imshow('eye', blk)
#-------------------
        # y = center_top[1]
        # x = left_point[0]
        # eye = frame[center_top[1]:center_bottom[1],left_point[0]:right_point[0]]
        # gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        # frame = roiDetector(gray_eye, frame)
        # cv2.imshow("roi", gray_eye)
#-------------------

    cv2.imshow("Frame", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()