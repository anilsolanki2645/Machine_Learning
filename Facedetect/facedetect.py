import cv2
import sys

images = cv2.imread("F:/ic812/AML/Facedetect/faces.jpg")
gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=3,minSize=(30,30))
print("[INFO] Found {0} Faces.".format(len(faces)))
for(x,y,w,h) in faces:
    cv2.rectangle(images,(x,y),(x+w,y+h),(0,255,0),2)
    roi_color = images[y:y+h,x:x+w]
    print("[INFO] Object found Saving locally.")
    cv2.imwrite(str(w)+str(h)+'_faces.jpg',roi_color)
status = cv2.imwrite('faces_detecte.jpg',images)
print("[INFO] Images Face_detected.jpg written to filesystem : ",status)
cv2.imshow('iamge',images)
cv2.waitKey()
