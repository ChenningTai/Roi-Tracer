import cv2
import numpy as np
import dlib
from math import hypot

#https://tpu.thinkpower.com.tw/tpu/articleDetails/950
#pip install CMake
#推薦:阿洲的城市教學
# close erode open
# cv2.morphologyEx(pupilFrame, )
#在function裡面的是區域變數，無法直接存取---已解決


def roiDetector(gray_Eye, frame, x, y):
    # ret, gray_Eye = video_capture.read()#boolean, gray_Eye
    # h, w = gray_Eye.shape
    roi = gray_Eye.copy()
    # rows, cols = roi.shape
    # cv2.imshow('v', roi)
    # cv2.imshow('roi', roi)
    # gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.equalizeHist(roi)
    roi = cv2.GaussianBlur(roi, (7 ,7), 0)
    # cv2.imshow('complete', roi)
    roi = cv2.equalizeHist(roi)
    _, threshold = cv2.threshold(roi, 40, 255, cv2.THRESH_BINARY_INV)#若無equalize:40
    kernal = np.ones((3, 3),np.uint8)
    # cv2.imshow('threshold_before dilate', threshold)
    threshold = cv2.dilate(threshold, kernal)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    # for cnt in contours: cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3) #draw contours
    ##-----------------from net, draw rec on Contours
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)###
    for cnt in contours:
        (xx, yy, ww, hh) = cv2.boundingRect(cnt)
       #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
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

    return frame, np.hstack([roi, threshold])

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
    frame, analysis = roiDetector(gray_eye, frame, x, y)
    # cv2.imshow("roi", gray_eye)
#-------------------
    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
    return np.hstack([analysis, gray_eye])


cap = cv2.VideoCapture(1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    blk = np.zeros((500,500), np.uint8)
    for face in faces:
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        landmarks = predictor(gray, face)
        analysisR = get_eye([36, 37, 38, 39, 40, 41], landmarks, frame)
        analysisL = get_eye([42, 43, 44, 45, 46, 47], landmarks, frame)
        # cv2.imshow("eye", np.hstack([analysisR, analysisL]))
        h1, w1 = analysisR.shape
        h2, w2 = analysisL.shape
        blk[0:h1, 0:w1] = analysisR
        blk[50:(h2+50), 0:w2] = analysisL
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