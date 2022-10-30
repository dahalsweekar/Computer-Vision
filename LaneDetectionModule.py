import cv2
import numpy as np
import utils

def getLaneCurve(vid):
    vidThres = utils.thresholding(vid)

    h,w,c = vid.shape
    points = utils.valTrackbars()
    vidWarp = utils.warpImg(vid,points,w,h)

    cv2.imshow("Thres",vidThres)
    cv2.imshow("Warp",vidWarp)
    return vidThres

def getCanny(vid):
    vidBlur = cv2.GaussianBlur(vid,(9,9),0)
    vidCanny = cv2.Canny(vidBlur,50,200)
    return vidCanny

def displayLine(vid, lines):
    line_image = np.zeros_like(vid)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),3)
    return line_image

cap = cv2.VideoCapture('Resources/Lane2.mp4')
frameCounter = 0
initialTrackBarVals = [100,100,100,100]
utils.InitializeTranckbars(initialTrackBarVals)

while True:
    frameCounter += 1
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0

    _, vid = cap.read()
    vid = cv2.resize(vid,(480,240))
    #getLaneCurve(vid)
    final_vid = getCanny(vid)
    lines = cv2.HoughLinesP(final_vid,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    line_vid = displayLine(final_vid,lines)
    cv2.imshow("Original",line_vid)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.waitKey(-1)