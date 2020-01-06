import cv2
import imutils
import numpy as np
from PIL import Image
import pytesseract
import fitz


def get_families(image):

    # convert the image to grayscale, find edges and dilates the image
    gray = get_grayscale(image)
    edged = cv2.Canny(gray, 100, 200)
    edged = do_dilation(edged)

    cv2.imshow("Gray - Canny - Dilate", edged)
    cv2.waitKey(0)

    binary = cv2.bitwise_not(edged)

    # find contours in the edged image
    cnts = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]
    
    
    areas = cv.contourArea(cnt)
    print(areas)
    
    screenCnt = None

    # loop over the contours
    for c in cnts:

        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.0005 * peri, True)

        # assume that we have found our screen
        x, y, width, height = cv2.boundingRect(c)
        family = image[y:y+height, x:x+width]
        cv2.imshow("family", family)
        cv2.waitKey(0)

        # draw the contours of the family in the main image
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
        cv2.imshow("image", image)
        cv2.waitKey(0)

        get_names(family)

        x, y, width, height = cv2.boundingRect(c)
        roi2 = edged[y:y+height, x:x+width]
        cv2.imshow("roi2", roi2)
        cv2.waitKey(0)

        x, y, width, height = cv2.boundingRect(c)
        roi3 = image[y:y+height, x:x+width]
        cv2.imshow("roi2", roi2)
        cv2.waitKey(0)

        cnts2 = cv2.findContours(
            roi2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts2 = imutils.grab_contours(cnts2)
        cnts2 = sorted(cnts2, key=cv2.contourArea, reverse=True)
        screenCnt2 = None

        # loop over our contours
        for c2 in cnts2:
            # approximate the contour
            peri2 = cv2.arcLength(c, True)
            approx2 = cv2.approxPolyDP(c, 0.0005 * peri2, True)

            # if our approximated contour has four points, then
            # we can assume that we have found our screen

            # print(approx2)
            screenCnt2 = approx2

            if len(approx2) == 4:
                screenCnt2 = approx2
                print('cuadrado')

            if len(approx2) == 3:
                screenCnt2 = approx2
                print('triangulo')

            x, y, width, height = cv2.boundingRect(c2)
            roi4 = roi2[y:y+height, x:x+width]
            cv2.imshow("Game Boy Screen", roi4)
            cv2.waitKey(0)

            x, y, width, height = cv2.boundingRect(c2)
            roi5 = roi3[y:y+height, x:x+width]
            cv2.imshow("Game Boy Screen", roi5)
            cv2.waitKey(0)


def get_names(I):

    # Params
    maxArea = 150
    minArea = 10

    _I = I.copy()

    Igray = get_grayscale(I)
    #erode = get_erode(Igray)
    #dilate = get_deskew(Igray)

    # Invert image to use for Tesseract
    #cv2.imshow('gray', Igray)
    #cv2.imshow('thersh', erode)
    #cv2.imshow('_opening', dilate)

    # Threshold
    ret, Ithresh = cv2.threshold(
        Igray, 175, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow(f'Ithresh', Ithresh)

    # Keep only small components but not to small
    comp = cv2.connectedComponentsWithStats(Ithresh)

    labels = comp[1]
    labelStats = comp[2]
    labelAreas = labelStats[:, 4]

    for compLabel in range(1, comp[0], 1):

        if labelAreas[compLabel] > maxArea or labelAreas[compLabel] < minArea:
            labels[labels == compLabel] = 0

    labels[labels > 0] = 1

    # Do dilation
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    IdilateText = cv2.morphologyEx(
        labels.astype(np.uint8), cv2.MORPH_DILATE, se)

    # Find connected component again
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        IdilateText)

    # Draw a rectangle around the text
    #labels = comp[1]
    labelStats = comp[2]
    #labelAreas = labelStats[:,4]

    for compLabel in range(1, num_labels, 1):

        cv2.rectangle(I, (stats[compLabel, 0], stats[compLabel, 1]), (stats[compLabel, 0] +
                                                                      stats[compLabel, 2], stats[compLabel, 1]+stats[compLabel, 3]), (0, 0, 255), 2)

        cv2.imshow("Biggest component", I)
        x = stats[compLabel, 0]
        y = stats[compLabel, 1]
        h = stats[compLabel, 3]
        w = stats[compLabel, 2]
        #print(x, y, h, w)

        #__I = get_grayscale(_I)
        delta = 10
        crop_img = _I[y+delta:y+h-delta, x+delta:x+w-delta]
        crop_img = cv2.resize(crop_img, None, fx=5, fy=5,
                              interpolation=cv2.INTER_CUBIC)
        crop_img, crop_img2 = ajust_contrast(crop_img)
        crop_img2 = remove_noise(crop_img2)

        cv2.imshow('crop_img', crop_img)
        cv2.imshow('crop_img2', crop_img2)

        #new_size = tuple(2*x for x in crop_img2.size)
        #crop_img2 = crop_img2.resize(new_size, Image.ANTIALIAS)
        crop_img2 = imutils.resize(crop_img2, width=800)
        print(pytesseract.pytesseract.image_to_string(crop_img2))
        cv2.waitKey()


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

# get grayscale image


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal


def remove_noise(image):
    return cv2.medianBlur(image, 5)

# thresholding


def get_thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# dilation


def do_dilation(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

# erosion


def get_erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)



#opening - erosion followed by dilation
def get_opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#opening - erosion followed by dilation
def get_ellipse(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_ELLIPSE, kernel)
# canny edge detection


def get_canny(image):
    return cv2.Canny(image, 5, 5)

# skew correction


def get_deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# template matching


def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


class ShapeDetector:


    def __init__(self):
        pass


    def detect(self, c):

        # initialize the shape name and approximate the contour
        shape = "unknown"
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)
        hull = cv2.convexHull(approx)

        area = cv2.contourArea(c)

        if len(approx) == len(hull):

            # if the shape is a triangle, it will have 3 vertices
            if len(approx) == 1:
                    shape = "line"
            elif len(approx) == 3:
                shape = "triangle"

            # if the shape has 4 vertices, it is either a square or
            # a rectangle
            elif len(approx) == 4:
                # compute the bounding box of the contour and use the
                # bounding box to compute the aspect ratio
                (x, y, w, h) = cv2.boundingRect(approx)
                ar = w / float(h)

                # a square will have an aspect ratio that is approximately
                # equal to one, otherwise, the shape is a rectangle
                shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

            # if the shape is a pentagon, it will have 5 vertices
            elif len(approx) == 5:
                shape = "pentagon"

            # otherwise, we assume the shape is a circle
            else:
                shape = "circle"

        # return the name of the shape
        return shape, area


def load_image_from_pdf():

    # Load pdf and get image from second page
    pdffile = u'D:\Dropbox\Python Projects\Rihal-Challenge\Rihal-Challenge.pdf'
    doc = fitz.open(pdffile)
    page = doc.loadPage(1)  # number of page
    zoom = 2    # zoom factor
    mat = fitz.Matrix(zoom, zoom)
    pix = page.getPixmap(matrix=mat, alpha=False)

    # Save image
    output = u"D:\Dropbox\Python Projects\Rihal-Challenge\src\images\out.png"
    pix.writePNG(output)

    # Load final image from file
    image = cv2.imread(
        u'D:\Dropbox\Python Projects\Rihal-Challenge\src\images\out.png')
    return image


def undesired_objects(image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        image, connectivity=4, ltype=cv2.CV_32S)
    print(centroids)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img3 = np.zeros(output.shape)
    for i in range(nb_components):
        img3[output == sizes[i]] = 255
        cv2.imshow("Each component", img3)
        cv2.waitKey()

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    cv2.imshow("Biggest component", img2)
    cv2.waitKey()


if __name__ == "__main__":

    url = u'D:\Dropbox\Python Projects\Rihal-Challenge\src\images\/test-b3.png'
    image = cv2.imread(url)
    gray = get_grayscale(image)
    ret, thersh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    blur = cv2.blur(thersh,(5,5))
    edged = get_canny(blur)
    cv2.imshow("thersh", edged)
    cv2.waitKey()

    blur = cv2.blur(edged,(10,10))    
    # find contours in the edged image
    cnts = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours, hierarchy = cnts
    print(len(hierarchy), hierarchy)

    
    total_area = sum([cv2.contourArea(contour) for contour in contours])
    print(total_area)

    for contour in contours:
        area = cv2.contourArea(contour)
        print(area, area *100 / total_area)
        

    modes = {'cv2.RETR_EXTERNAL': cv2.RETR_EXTERNAL,
             'cv2.RETR_CCOMP': cv2.RETR_CCOMP,
             'cv2.RETR_LIST': cv2.RETR_LIST,
             'cv2.RETR_TREE': cv2.RETR_TREE}

    for k, v in modes.items():
        contours, hierarchy = cv2.findContours(
            edged, v, cv2.CHAIN_APPROX_SIMPLE)
        # print(k, hierarchy, end='\n\n')

    print(hierarchy)

    cnts = imutils.grab_contours(cnts)
    #cnts = sorted(cnts, key=cv2.contourArea, reverse=True)



    families = []

    # loop over the contours
    for idx, c in enumerate(cnts):

        if hierarchy[0][idx][1] == -1:  # start new family
            family = {}
        else:
        
            sd = ShapeDetector()
            print(sd.detect(c))

            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.0005 * peri, True)

            # assume that we have found our screen
            x, y, width, height = cv2.boundingRect(c)
            family = image[y:y+height, x:x+width]
            cv2.imshow("family", family)
            cv2.waitKey(0)

            print(hierarchy[0][idx])

            # draw the contours of the family in the main image
            cv2.drawContours(image, [approx], -1, (255, 255, 0), 3)
            cv2.imshow("image", image)
            cv2.waitKey(0)
