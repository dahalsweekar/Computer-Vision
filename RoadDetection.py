import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('Resources/Road.avi')

def vidProcess(vid):
    vidCopy = cv2.GaussianBlur(vid,(5,5),0)
    vidHSV = cv2.cvtColor(vidCopy,cv2.COLOR_BGR2HSV)
    return vidHSV

def drawContour(vid):
    contours, hierarchy = cv2.findContours(vid,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area>500:
            cv2.drawContours(vidCopy,cnt,-1,(255,0,0),2)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            x,y,w,h = cv2.boundingRect(approx)

def polyMask(vid):
    traingle = np.array([(0,350),(700,350),(300,200)])
    mask = np.zeros_like(vid)
    cv2.fillPoly(mask,[traingle],(255,255,255))
    masked_vid = cv2.bitwise_and(vid,mask)
    return masked_vid


while True:
    success, vid = cap.read()
    vid = cv2.resize(vid,(640,480))
    vidCopy = vid.copy()
    vidMasked = polyMask(vidCopy)
    vidGray = cv2.cvtColor(vidMasked,cv2.COLOR_BGR2GRAY)
    vidHSV = vidProcess(vidMasked)
    h_min = 0
    h_max = 179
    s_min = 30
    s_max = 255
    v_min = 0
    v_max = 255
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(vidHSV,lower,upper)
    vidResult = cv2.bitwise_or(vid,vid,mask=mask)
    drawContour(mask)
    cv2.imshow("Road Masked",vidCopy)
    cv2.imshow("Road Video", mask)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.waitKey(-1)
