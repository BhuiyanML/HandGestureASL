###################################################################
################ Real-time Hand Gesture Recognition ###############
################ Computer Vision Course Project ###################
################ Rasel Ahmed Bhuiyan ##############################
################ PhD Student, University of Notre Dame ############
################ Email: rbhuiyan@nd.edu ###########################
################ Fall Semester --- September 2022 #################

import cv2
import torch
from model import ConvNet
from cvzone.HandTrackingModule import HandDetector
import torchvision.transforms as transforms
from PIL import Image

# Assigning device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Connecting Webcam
cap = cv2.VideoCapture(0)

# Hand tracker object
detector = HandDetector(maxHands=1)

# Hyperperameters
offset = 30
num_class = 3
cropSize = 224
labels = ['A', 'B', 'C']

# Data pre-processor
test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(cropSize),
        transforms.ToTensor()
    ])

# Classifier
model = ConvNet(num_class=num_class).to(device)
model.load_state_dict(torch.load("ASL-Weights/best_model.pth"))
model.eval()

# Classify the real time hend gesture
while True:
    # Reading video frame from webcam
    success, img = cap.read()
    imageOutput = img.copy()
    image = img.copy()
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
        cropImg = image[y-offset:y+h+offset, x-offset:x+w+offset]

        # Convert cropped image into PIL image to pass into the transformers
        testImage = cv2.cvtColor(cropImg, cv2.COLOR_BGR2RGB)
        testImage = Image.fromarray(testImage)
        testImage = test_transform(testImage)
        testImage = testImage.unsqueeze(0)
        testImage = testImage.to(device)
        # Classify the test image
        outputs = model(testImage)
        print(outputs)
        _, index = torch.max(outputs, 1)
        index = index.detach().cpu().numpy()

        # Draw rectangle for showing text
        cv2.rectangle(imageOutput, (x-20, y-110), (x+80, y-20),
                      color=(255, 0, 255), thickness=cv2.FILLED)
        # Put text into the box
        cv2.putText(imageOutput, labels[index[0]], (x, y-35), cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=3, thickness=2, color=(255, 255, 255))

        # Draw rectangle around the hand
        cv2.rectangle(imageOutput, (x-20, y-20), (x+w+20, y+h+20),
                      color=(255, 0, 255), thickness=4)

        # # Text to speech
        # text_to_speech(labels[index[0]])

    cv2.imshow("Hand Gesture", imageOutput)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

