import cv2
import numpy as np
import face_recognition

imgMe = face_recognition.load_image_file('Resources/Me1.jpg')
imgMe=cv2.resize(imgMe,(400,400))
imgMe = cv2.cvtColor(imgMe,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Resources/Me3.jpg')
imgTest = cv2.resize(imgTest,(400,400))
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLocation = face_recognition.face_locations(imgMe)[0]
encodeMe = face_recognition.face_encodings(imgMe)[0]
cv2.rectangle(imgMe,(faceLocation[3],faceLocation[0]),(faceLocation[1],faceLocation[2]),(255,0,0),2)

faceLocationTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocationTest[3],faceLocationTest[0]),(faceLocationTest[1],faceLocationTest[2]),(0,255,0),2)

results = face_recognition.compare_faces([encodeMe],encodeTest)
faceDis = face_recognition.face_distance([encodeMe],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow("Me",imgMe)
cv2.imshow("Test",imgTest)
cv2.waitKey(0)