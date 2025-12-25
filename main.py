import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

amrit_face = face_recognition.load_image_file("/Users/amritkumaryadav/Desktop/New/Devlopment in Python/FaceRecognition/faces/amrit.jpg")
amrit_encoding   = face_recognition.face_encodings(amrit_face)[0]

ram_face = face_recognition.load_image_file("/Users/amritkumaryadav/Desktop/New/Devlopment in Python/FaceRecognition/faces/ram.jpg")
ram_encoding   = face_recognition.face_encodings(ram_face)[0]

shyaam_face = face_recognition.load_image_file("/Users/amritkumaryadav/Desktop/New/Devlopment in Python/FaceRecognition/faces/shyaam.jpg")
shyaam_encoding   = face_recognition.face_encodings(shyaam_face)[0]

lalit_face = face_recognition.load_image_file("/Users/amritkumaryadav/Desktop/New/Devlopment in Python/FaceRecognition/faces/lalit.jpg")
lalit_encoding   = face_recognition.face_encodings(lalit_face)[0]

known_face_encoding = [amrit_encoding,ram_encoding,shyaam_encoding,lalit_encoding]
known_face_name = ["amrit","ram","shyaam","lalit"]

students = known_face_name.copy()

face_location = []
face_encoding = []

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv","w+",newline="")
lnwriter = csv.writer(f)

while True: 
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rbg_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_RGB2BGR)
    
    
    face_locations = face_recognition.face_locations(rbg_small_frame)
    face_encodings = face_recognition.face_encodings(rbg_small_frame,face_locations)
    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
        face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
        best_match_index = np.argmin(face_distance)
        
        if(matches[best_match_index]):
            name = known_face_name[best_match_index]
            
        # add text if a person is present 
        if name in known_face_name:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10,100)
            fontScale = 1.5
            fontColor = (255,0,0)
            thickness = 3
            lineType = 2
            cv2.putText(frame,name + " Present", bottomLeftCornerOfText,font,fontScale,fontColor,thickness,lineType)
            
            if name in students:
                students.remove(name)
                current_time = now.strftime("%H:%M:%S")
                lnwriter.writerow([name,current_time])
    cv2.imshow("Attendence",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()