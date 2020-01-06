import cv2
import numpy as np


cap = cv2.VideoCapture(0)

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


while True:
    _, frame = cap.read()
    image = frame.copy()
    #image = cv2.imread('1.jpg')
    result = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #lower = np.array([130,25,0])
    #upper = np.array([179,255,255])

    #lower red
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])

    #upper red
    lower_red2 = np.array([15,50,50])
    upper_red2 = np.array([180,255,255])

    mask1 = cv2.inRange(image, lower_red, upper_red)
    mask2 = cv2.inRange(image, lower_red2, upper_red2)

    mask = mask1 + mask2

    result = cv2.bitwise_and(result, result, mask=mask)

    result = get_grayscale(result)
    cnts = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    c = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
    x, y, width, height = cv2.boundingRect(c)
    family = image[y:y+height, x:x+width]
    
    cv2.imshow('mask', mask)
    cv2.imshow('result', result)
    cv2.imshow('family', family)

    key = cv2.waitKey(1)
    if key == 27:
        break
    """
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([170, 120, 70])
    upper_blue = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    key = cv2.waitKey(1)
    if key == 27:
        break
    """

cap.release()
