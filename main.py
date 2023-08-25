import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

# Initialize the hand detector with a maximum of one hand detection
detector = HandDetector(maxHands=1)

# Offset for cropping around the detected hand
offset = 20

# Desired size of the cropped hand image
imgSize = 300

# Folder path to save captured images
folder = "Data/C"

# Counter to keep track of saved images
counter = 0

while True:
    # Capture a frame from the webcam
    success, img = cap.read()

    # Detect hands in the captured frame
    hands, img = detector.findHands(img)

    if hands:
        # Get information about the first detected hand
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white background image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the captured frame around the hand
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Calculate aspect ratio of the cropped region
        aspectRatio = h / w

        if aspectRatio > 1:
            # Resize the cropped region with a vertical orientation
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            # Resize the cropped region with a horizontal orientation
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Display the cropped hand image
        cv2.imshow("ImageCrop", imgCrop)

        # Display the composite image with white background
        cv2.imshow("ImageWhite", imgWhite)

    # Display the original captured frame with detected hand bounding box
    cv2.imshow("Image", img)

    # Check for key press event
    key = cv2.waitKey(1)

    if key == ord("s"):
        # Save the composite image with a timestamped filename
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
