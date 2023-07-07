import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm


###############################
brushThickness = 15
eraserThickness = 100

##############################



folderPath = "Header"
myList = os.listdir(folderPath)
myList = [file for file in myList if file != '.DS_Store']  # Exclude .DS_Store
myList = list(reversed(myList))
myList = sorted(myList, key=lambda x: int(x.split('.')[0]))
print(myList)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
# header = cv2.resize(header, (1280, 125))  # Resize header image to match assigned region
header = cv2.resize(header, (1280, 62))
drawColor = (255, 0, 255)


cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width, height)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionConf=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:

    # 1. Import Image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img )
    lmList = detector.findPosition(img, draw=False)

    if len(lmList)!=0:

        # print(lmList)

        # tip of the ndex and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # 4. If selection mode - Two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection Mode")
            # Checking for the click
            if y1 < 125:
                if 170< x1 < 360:
                    header = overlayList[0]
                    drawColor = (0, 255, 255)
                elif 470 < x1 < 600:
                    header = overlayList[1]
                    drawColor = (0xFF, 0xFF, 0x80)
                elif 650 < x1 < 850:
                    header = overlayList[2]
                    drawColor = (255, 0, 255)
                elif 970 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. If drawing mode - Index finger is up
        if fingers[1] and fingers[2]  == False:

            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

#################################


    # Setting the header image
    header_resized = cv2.resize(header, (img.shape[1], 125))
    img[0:125, 0:img.shape[1]] = header_resized
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)
