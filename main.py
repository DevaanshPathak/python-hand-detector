# Import necessary libraries
from cvzone.HandTrackingModule import HandDetector  # Import the HandDetector class from cvzone module
import cv2  # Import OpenCV library

# Create a VideoCapture object to capture video from the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Create an instance of the HandDetector class with specified detection confidence and maximum hands to detect
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Infinite loop to continuously process frames from the camera
while True:
    # Read a frame from the video capture
    success, img = cap.read()

    # Find hands and their landmarks in the current frame
    hands, img = detector.findHands(img)  # Detect and draw hand landmarks
    
    # Check if hands are detected
    if hands:
        # Process information for the first hand
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info: x, y, width, height
        centerPoint1 = hand1['center']  # Center of the hand: cx, cy
        handType1 = hand1["type"]  # Hand type: "Left" or "Right"
        fingers1 = detector.fingersUp(hand1)  # Get finger status
        
        # If there are two detected hands
        if len(hands) == 2:
            # Process information for the second hand
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Landmark points
            bbox2 = hand2["bbox"]  # Bounding box info: x, y, width, height
            centerPoint2 = hand2['center']  # Center of the hand: cx, cy
            handType2 = hand2["type"]  # Hand type: "Left" or "Right"
            fingers2 = detector.fingersUp(hand2)  # Get finger status

            # Find the distance between two specific landmarks on different hands
            length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)  # Detect and draw distance

    # Display the modified image with detected hands, landmarks, and information
    cv2.imshow("Image", img)
    cv2.waitKey(1)

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()