import cv2
import numpy as np
import dlib
from math import hypot
import time
#https://tpu.thinkpower.com.tw/tpu/articleDetails/950
#pip install CMake
#推薦:阿洲的城市教學
# close erode open  #現在可考慮close
# cv2.morphologyEx(pupilFrame, )
#在function裡面的是區域變數，無法直接存取---已解決
#把二質化後的各處理成果顯示
#十張相片取平均位置
#眼睛高度優化
    #考慮用雙重:瞳孔位置、眼睛高度判斷   實驗:用拍攝的四張相片分析#目前看起來還可行，先暫放
#可考慮用二維線性映射  #重要性在於螢幕並非完全水平時



def roiDetector(gray_Eye, frame, x, y):
    # ret, gray_Eye = video_capture.read()#boolean, gray_Eye
    h, _ = gray_Eye.shape #為了側上下視線。 與下面那行將來合併
    roi = gray_Eye.copy()
    # rows, cols = roi.shape
    # cv2.imshow('v', roi)
    # cv2.imshow('roi', roi)
    # gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi0 = cv2.equalizeHist(roi)
    # roi1 = roi0
    roi1 = cv2.GaussianBlur(roi0, (7 ,7), 0)
    # cv2.imshow('complete', roi)
    roi = cv2.equalizeHist(roi1)
    _, threshold = cv2.threshold(roi, 40, 255, cv2.THRESH_BINARY_INV)#若無equalize:40
    kernal = np.ones((2, 2),np.uint8)
    # cv2.imshow('threshold_before dilate', threshold)
    # threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernal)
    threshold = cv2.dilate(threshold, kernal)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    # for cnt in contours: cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3) #draw contours
    ##-----------------from net, draw rec on Contours
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)###
    for cnt in contours:
        (xx, yy, ww, hh) = cv2.boundingRect(cnt)
       #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
        centerPosi_x = xx + ww/2
        # centerPosi_y = yy + hh/2
        centerPosi_y = h #可在前面賦予h值處就直接推到centerPosi_y
        xx = xx+x
        yy = yy+y-5
        cv2.rectangle(frame, (xx, yy), (xx + ww, yy + hh), (0, 0, 255), 2)
        # cv2.line(frame, (xx + int(ww/2), 0), (xx + int(ww/2), rows), (0, 255, 0), 2)
        # cv2.line(frame, (0, yy + int(hh/2)), (cols, yy + int(hh/2)), (0, 255, 0), 2)
        break
# # method 2-------------------------
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

def get_eye(eye_points, facial_landmarks,frame):
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
#計算瞳孔位置比例
    widRatio = centerPosi_x / (right_point[0]-x)
    # heiRatio = centerPosi_y / (center_bottom[1]-y+5)#原本判定視線高度方法
    heiRatio = centerPosi_y#將來要直接賦值(方案最後整理時處理)
#-------------------
    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
    return np.hstack([gray_eye, analysis]), [widRatio, heiRatio]


def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def Pointer(ratio):#傳回座標[[右眼寬,右眼高], [左眼寬,左眼高]]
#先右後左
    # Left = [0.75,0.7]
    # WidCenter = [0.6, 0.5]
    # Right = [0.43, 0.3]
    # Up = [14,14]
    # Down = [9,9]
    Left = [upLeft[0],upLeft[2]]
    Right = [downRight[0], downRight[2]]
    Up = [upLeft[1], upLeft[3]]
    Down = [downRight[1], downRight[3]]

    x = [-1,-1]
    y = [-1,-1]
    for i in range(2):
        x[i] = (ratio[i][0]-Left[i])/(Right[i]-Left[i])
        y[i] = (ratio[i][1]-Up[i])/(Down[i]-Up[i])
    return int(sum(x)*1000), int(sum(y)*500)#兩眼資料取平均(這樣是最有效的嗎??)
#將原公式*2000/2 改寫成*1000，*500抑是
def Correction():
    posiList = [[0.5]*64,[0.5]*64,[0.5]*64,[0.5]*64]#若使用[[0.5]*32]*4則會錯將外層[]中四個元素指向同一個參考位置
    # i = 1
    mean = []
    for i in range(64):
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            #x, y = face.left(), face.top()
            #x1, y1 = face.right(), face.bottom()
            #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            landmarks = predictor(gray, face)
            centerRatio = [None, None]
            # centerRatioY = [None, None]
            analysisR, centerRatio[0] = get_eye([36, 37, 38, 39, 40, 41], landmarks, frame)#這邊讀取傳回直應該將左右分法改成xy分法
            analysisL, centerRatio[1] = get_eye([42, 43, 44, 45, 46, 47], landmarks, frame)
            for n in range(2):
                for m in range(2):
                    k = n*2+m
                    posiList[k][i] = centerRatio[n][m]#rx
    for each in posiList:
        each.sort()
        mean.append(sum(each[17:48])/32)
    return mean


#Main():
# cap = cv2.VideoCapture('news.mp4')
cap = cv2.VideoCapture(1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#Correction
print('Start 1...')
time.sleep(1)
upLeft = Correction()
print('Start 2...')
time.sleep(1)
downRight = Correction()
#prepare
ini = 0.5
eightFotoX = [ini]*8#for Purify()
eightFotoY = [ini]*8#for Purify()
i = 0
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    blk = np.zeros((1000,2000), np.uint8)
    for face in faces:
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        landmarks = predictor(gray, face)
        centerRatio = [None, None]
        # centerRatioY = [None, None]
        analysisR, centerRatio[0] = get_eye([36, 37, 38, 39, 40, 41], landmarks, frame)#這邊讀取傳回直應該將左右分法改成xy分法
        analysisL, centerRatio[1] = get_eye([42, 43, 44, 45, 46, 47], landmarks, frame)
        sightPoint_X, sightPoint_Y = Pointer(centerRatio)
        #Purification():
        j = i%8
        # print(j, type(eightFotoX[1]))
        eightFotoX[j] = sightPoint_X#兩眼資料取平均(這樣是最有效的嗎??) 
        eightFotoX_u = int(sum(eightFotoX)/8)
        eightFotoY[j] = sightPoint_Y #兩眼資料取平均(這樣是最有效的嗎??)
        eightFotoY_u = int(sum(eightFotoY)/8)
        i += 1

#analysis
        h1, w1 = analysisR.shape
        h2, w2 = analysisL.shape
        # cv2.circle(blk, (600,300), 10, (255,0,0), 2)
        # cv2.circle(blk, (100,300), 10, (255,0,0), 2)
        blk[0:h1, 0:w1] = analysisR
        blk[50:(h2+50), 0:w2] = analysisL
        cv2.circle(blk, (eightFotoX_u,eightFotoY_u), 10, (255,0,0), 2) #, lineType=None, shift=None)
        # cv2.circle(blk, (sightPoint_X,sightPoint_Y), 10, (255,0,0), -1) #, lineType=None, shift=None)
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