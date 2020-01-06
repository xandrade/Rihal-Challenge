import cv2
import imutils
import numpy as np
from PIL import Image
import pytesseract
import fitz

# Params
maxArea = 150
minArea = 10

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def get_thresholding(image):
    return cv2.threshold(image, 170, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def get_erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


def do_dilation(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def get_canny(image):
    return cv2.Canny(image, 100, 200)

def remove_noise(image):
    return cv2.medianBlur(image, 5)

def ajust_contrast(img):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))

    # convert from BGR to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2, a, b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return img, img2


if __name__ == "__main__":

    url = u'D:\Dropbox\Python Projects\Rihal-Challenge\src\images\holy.jpeg'
    image = cv2.imread(url)

    gray = get_grayscale(image)

    #image = imutils.resize(image, width=500)
    


    #gray = get_thresholding(gray)

    #gray = imutils.resize(gray, width=1500)
    #print(pytesseract.pytesseract.image_to_string(gray))
    cv2.imshow(f'gray', gray)
    cv2.waitKey()

    # Threshold
    ret, Ithresh = cv2.threshold(gray,255,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow(f'Ithresh', Ithresh)
    cv2.waitKey()

    gray = get_erode(gray)
    gray = do_dilation(gray)

    cv2.imshow(f'gray', gray)
    cv2.waitKey()
    """
    Ithresh = get_canny(Ithresh)
    cv2.imshow(f'get_canny', Ithresh)
    cv2.waitKey()
    
    Ithresh = do_dilation(Ithresh)
    cv2.imshow(f'do_dilation', Ithresh)
    cv2.waitKey()
    """

    # Keep only small components but not to small
    comp = cv2.connectedComponentsWithStats(Ithresh)

    labels = comp[1]
    labelStats = comp[2]
    labelAreas = labelStats[:,4]

    for compLabel in range(1,comp[0],1):

        if labelAreas[compLabel] > maxArea or labelAreas[compLabel] < minArea:
            labels[labels==compLabel] = 0
    
    labels[labels>0] =  1


    # Do dilation
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    IdilateText = cv2.morphologyEx(labels.astype(np.uint8),cv2.MORPH_DILATE,se)

    # Find connected component again
    num_labels , labels, stats, centroids = cv2.connectedComponentsWithStats(IdilateText)

    # Draw a rectangle around the text
    #labels = comp[1]
    labelStats = comp[2]
    #labelAreas = labelStats[:,4]

    print(num_labels)

    for compLabel in range(1,num_labels,1):

        cv2.rectangle(image,(stats[compLabel,0],stats[compLabel,1]),(stats[compLabel,0]+stats[compLabel,2],stats[compLabel,1]+stats[compLabel,3]),(0,0,255),2)

        cv2.imshow("Biggest component", image)
        x = stats[compLabel,0]
        y = stats[compLabel,1]
        h = stats[compLabel,3]
        w = stats[compLabel,2]
        #print(x, y, h, w)

        #__I = get_grayscale(_I)
        crop_img = get_grayscale(image).copy()[y:y+h, x:x+w]
        #crop_img = cv2.resize(crop_img.copy(), None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        crop_img, crop_img2 = ajust_contrast(crop_img)
        crop_img2 = remove_noise(crop_img2)
        
        #cv2.imshow('crop_img', crop_img)
        #cv2.imshow('crop_img2', crop_img2)

        #new_size = tuple(2*x for x in crop_img2.size)
        crop_img2 = crop_img.copy()
        crop_img2 = cv2.bitwise_not(crop_img2)
        #crop_img2 = crop_img2.resize(new_size, Image.ANTIALIAS)
        #crop_img2 = imutils.resize(crop_img2, width=800)
        #ret, crop_img2 = cv2.threshold(crop_img2,200,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #cv2.imshow('crop_img', crop_img2)
        print(pytesseract.pytesseract.image_to_string(crop_img2))
        cv2.waitKey()