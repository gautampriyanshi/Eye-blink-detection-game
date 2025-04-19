import numpy as np
import cv2
import sys

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

first_read = True

cap = cv2.VideoCapture(0)

def display_message(img, text, position=(100, 100), color=(0, 0, 255), size=2):
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, size, color, 3, cv2.LINE_AA)

while True:
    ret, img = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 1, 1)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))
    
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_face = gray[y:y + h, x:x + w]
            roi_face_clr = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(50, 50))
            
            if len(eyes) >= 2:
                if first_read:
                    display_message(img, "Press 'S' to start", (50, 50), (0, 255, 255), 1.5)
                    
                else:
                    display_message(img, "Eyes open!", (50, 50), (0, 255, 255), 1.5)
                    print("----------------------")
                
            else:
                if first_read:
                    display_message(img, "No eyes detected", (50, 50), (0, 0, 255), 1.5)
                else:
                    print("You lose! Exiting game...")
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit()
    else:
        display_message(img, "No face detected", (50, 50), (255, 0, 0), 1.5)
    
    display_message(img, "Press 'Q' to exit", (10, img.shape[0] - 20), (255, 255, 255), 1)
    
    cv2.imshow('Eye Blink Game', img)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s') and first_read:
        first_read = False

cap.release()
cv2.destroyAllWindows()
