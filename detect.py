import cv2
import numpy as np
from scipy.spatial import distance as dist
import pygame

def eye_aspect_ratio(eye):
    if len(eye) != 6:
        return 0
    
    # compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    
    # compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    
    return ear

# load the Haar cascade classifiers for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
left_eye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')  # Path to left eye cascade XML file
right_eye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')  # Path to right eye cascade XML file

# define constants for EAR threshold and consecutive frame count
EAR_THRESHOLD = 0.3
EAR_CONSEC_FRAMES = 48

# initialize the frame counter and the total number of blinks
COUNTER = 0
alarm_playing = False  # Flag to track if alarm sound is currently playing

# Initialize pygame for sound
pygame.init()
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('alarm.wav')  # Replace 'alarm.wav' with your alarm sound file

# start the video stream
cap = cv2.VideoCapture(0)

while True:
    # grab the frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break
    
    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # loop over the face detections
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # detect left eye within the face region
        left_eyes = left_eye_cascade.detectMultiScale(roi_gray)
        
        # detect right eye within the face region
        right_eyes = right_eye_cascade.detectMultiScale(roi_gray)
        
        # ensure both left and right eyes are detected
        if len(left_eyes) == 1 and len(right_eyes) == 1:
            (lx, ly, lw, lh) = left_eyes[0]
            (rx, ry, rw, rh) = right_eyes[0]
            
            left_eye = roi_color[ly:ly+lh, lx:lx+lw]
            right_eye = roi_color[ry:ry+rh, rx:rx+rw]
            
            # calculate the eye aspect ratio for both eyes
            left_eye_gray = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
            right_eye_gray = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
            
            _, left_thresh = cv2.threshold(left_eye_gray, 70, 255, cv2.THRESH_BINARY)
            _, right_thresh = cv2.threshold(right_eye_gray, 70, 255, cv2.THRESH_BINARY)
            
            left_contours, _ = cv2.findContours(left_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            right_contours, _ = cv2.findContours(right_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if left_contours and right_contours:
                left_eye_hull = max(left_contours, key=cv2.contourArea)
                right_eye_hull = max(right_contours, key=cv2.contourArea)
                
                left_eye_points = np.array([pt[0] for pt in left_eye_hull])
                right_eye_points = np.array([pt[0] for pt in right_eye_hull])
                
                leftEAR = eye_aspect_ratio(left_eye_points)
                rightEAR = eye_aspect_ratio(right_eye_points)
                
                ear = (leftEAR + rightEAR) / 2.0
                
                # check if the eye aspect ratio is below the blink threshold, and if so, increment the blink frame counter
                if ear < EAR_THRESHOLD:
                    COUNTER += 1
                    
                    # if the eyes were closed for a sufficient number of frames, then sound an alarm
                    if COUNTER >= EAR_CONSEC_FRAMES and not alarm_playing:
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # Play alarm sound
                        alarm_sound.play()
                        alarm_playing = True
                else:
                    # reset the eye frame counter
                    COUNTER = 0
                    
                    # If alarm is currently playing and eyes are open, stop the alarm
                    if alarm_playing:
                        alarm_sound.stop()
                        alarm_playing = False
                
                # draw the total number of blinks on the frame along with the computed eye aspect ratio for the frame
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # show the frame
    cv2.imshow("Frame", frame)
    
    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()  # Quit pygame mixer
pygame.quit()  # Quit pygame
