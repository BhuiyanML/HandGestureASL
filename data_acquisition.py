###################################################################
################ Real-time Hand Gesture Recognition ###############
################ Computer Vision Course Project ###################
################ Rasel Ahmed Bhuiyan ##############################
################ PhD Student, University of Notre Dame ############
################ Email: rbhuiyan@nd.edu ###########################
################ Fall Semester --- September 2022 #################

import math
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Data Path
folder = 'handGestureData/val/C/'
# Counter for maintaining data serial
count = 0
# Padding for cropping image
# So that, we can get some spaces
# around the detected hand
offset = 30
# Original Image Size
imageSize = 300
# Connecting Webcams
cap = cv2.VideoCapture(0)
# Hand tracker object
detector = HandDetector(maxHands=1)

while True:
    # Reading video frame from webcam
    success, img = cap.read()
    Image = img.copy()
    # Detecting hand from static video frame
    hands, img = detector.findHands(img)

    # Checking hand found or not
    if hands:
        # Getting single hand information
        hand = hands[0]
        # Get hand type
        handType = hand['type']
        # Getting bounding box coordinates
        x, y, w, h = hand['bbox']
        # Cropping hand from original image
        cropImg = Image[y-offset:y+h+offset, x-offset:x+w+offset]

        cv2.imshow("Crop Image", cropImg)

    cv2.imshow("Original Image", img)

    key = cv2.waitKey(1)
    # Press s for saving data
    if key == ord('s'):
        count += 1
        cv2.imwrite(f'{folder}{handType}_hand_{count}.png', cropImg)
        print(count)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
