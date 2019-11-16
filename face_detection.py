#import libraries of python opencv
import opencv as cv2
import numpy as np
import pafy
import youtube_dl
import face_recognition as fr




#Create a VideoCapture object and read from input file
url = 'https://www.youtube.com/watch?v=YWcu_8xPSs8'
vPafy = pafy.new(url)
vid = vPafy.getbest(preftype="webm")
cap = cv2.VideoCapture(vid.url)

#Use trained xml classifier for face detection in the video
face_cascade = cv2.CascadeClassifier('G:/video lectures/djangofile/venv/tmtbars/Face Detection/haarcascade_frontalface_default.xml')

#Read until video is completed
while(True):
    #Capture frame by frame
    ret, frame = cap.read()
    #Convert the video into gray video without color
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Detect faces,eyes and mouth in the video
    faces = face_cascade.detectMultiScale(gray, 1.32, 5)
    #Draw a rectangle boxes around the faces
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
            
    #Display the resulting frame
    cv2.imshow('Frame', frame)

    #Press Q on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#when everything done, release the videocapture object
cap.release()
#closes all the frames
cv2.destroyAllWindows()
