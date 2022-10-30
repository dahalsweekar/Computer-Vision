import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


def landmarks():
    colorR = 255, 0, 255
    lmList = []
    xx,yy,ww,hh = 350,350,50,50
    if results.multi_hand_landmarks:
        for handlandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])
            mpDraw.draw_landmarks(img, handlandmark,mpHands.HAND_CONNECTIONS)
            if lmList:
                cursor = lmList[8]
                if xx-ww//2< cursor[1] < xx+ww//2 and xx-hh//2 < cursor[2] < xx+hh//2:
                    colorR = (0,255,0)
                    xx,yy = cursor[1],cursor[2]
    cv2.rectangle(img, (xx-ww//2, xx-hh//2), (xx+ww//2, xx+hh//2), colorR, cv2.FILLED)
    


while True:
    success,img = cap.read()
    img = cv2.flip(img,1)
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #cv2.rectangle(img, (350, 350), (400, 400), colorR, cv2.FILLED)
    results = hands.process(imgRGB)
    landmarks()
    cv2.imshow("Hand Gesture",img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
