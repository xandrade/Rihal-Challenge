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
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:3]
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

        cnts2 = cv2.findContours(roi2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts2 = imutils.grab_contours(cnts2)
        cnts2 = sorted(cnts2, key = cv2.contourArea, reverse = True)
        screenCnt2 = None

        # loop over our contours
        for c2 in cnts2:
            # approximate the contour
            peri2 = cv2.arcLength(c, True)
            approx2 = cv2.approxPolyDP(c, 0.0005 * peri2, True)
        
            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            
            #print(approx2)
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
    ret, Ithresh = cv2.threshold(Igray,175,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow(f'Ithresh', Ithresh)

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
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
    IdilateText = cv2.morphologyEx(labels.astype(np.uint8),cv2.MORPH_DILATE,se)

    # Find connected component again
    num_labels , labels, stats, centroids = cv2.connectedComponentsWithStats(IdilateText)

    # Draw a rectangle around the text
    #labels = comp[1]
    labelStats = comp[2]
    #labelAreas = labelStats[:,4]

    for compLabel in range(1,num_labels,1):

        cv2.rectangle(I,(stats[compLabel,0],stats[compLabel,1]),(stats[compLabel,0]+stats[compLabel,2],stats[compLabel,1]+stats[compLabel,3]),(0,0,255),2)

        cv2.imshow("Biggest component", I)
        x = stats[compLabel,0]
        y = stats[compLabel,1]
        h = stats[compLabel,3]
        w = stats[compLabel,2]
        #print(x, y, h, w)

        #__I = get_grayscale(_I)
        delta = 10
        crop_img = _I[y+delta:y+h-delta, x+delta:x+w-delta]
        crop_img = cv2.resize(crop_img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
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
    clahe=cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

    lab=cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l,a,b=cv2.split(lab)  # split on 3 different channels

    l2=clahe.apply(l)  # apply CLAHE to the L-channel

    lab=cv2.merge((l2,a,b))  # merge channels
    img2=cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return img,img2

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def get_thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def do_dilation(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def get_erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def get_opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#opening - erosion followed by dilation
def get_ellipse(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_ELLIPSE, kernel)

#canny edge detection
def get_canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
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
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
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
		return shape


def load_image_from_pdf():

    # Load pdf and get image from second page
    pdffile = u'D:\Dropbox\Python Projects\Rihal-Challenge\Rihal-Challenge.pdf'
    doc = fitz.open(pdffile)
    page = doc.loadPage(1) #number of page
    zoom = 2    # zoom factor
    mat = fitz.Matrix(zoom, zoom)
    pix = page.getPixmap(matrix = mat, alpha=False)
    
    # Save image 
    output = u"D:\Dropbox\Python Projects\Rihal-Challenge\src\images\out.png"
    pix.writePNG(output)
    
    # Load final image from file
    image = cv2.imread(u'D:\Dropbox\Python Projects\Rihal-Challenge\src\images\out.png')
    return image


 
def undesired_objects (image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4, ltype=cv2.CV_32S)
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

    
    image = load_image_from_pdf()
    get_families(image)

    img = cv2.imread(u'D:\Dropbox\Python Projects\Rihal-Challenge\src\images\sample1.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thersh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    cv2.imshow("thersh", thersh)
    cv2.waitKey()
    
    comp = cv2.connectedComponentsWithStats(thersh)
    
    undesired_objects(thersh)

    # dst = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)



    #text = pytesseract.pytesseract.image_to_string(Image.open(u'D:\Dropbox\Python Projects\Rihal-Challenge\src\images\sample1.jpg'))
    #print(text)

    # load the image and resize it to a smaller factor so that
    # the shapes can be approximated better
    image = cv2.imread(u'D:\Dropbox\Python Projects\Rihal-Challenge\src\images\sample2.jpg')
    output = image.copy()
    resized = imutils.resize(image, width=800)
    resized = image.copy()
    ratio = image.shape[0] / float(resized.shape[0])



    img = cv2.imread(u'D:\Dropbox\Python Projects\Rihal-Challenge\src\images\sample1.jpg')
    mask = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)[1][:,:,0]
    mask = cv2.imread(u'D:\Dropbox\Python Projects\Rihal-Challenge\src\images\sample1.jpg', 0)
    dst = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)

    cv2.imshow("mask", mask)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)



    font = cv2.FONT_HERSHEY_COMPLEX
    img = cv2.imread(u'D:\Dropbox\Python Projects\Rihal-Challenge\src\images\sample1.jpg', cv2.IMREAD_GRAYSCALE)


    text = pytesseract.pytesseract.image_to_string(img)
    print(text)

    _, threshold = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)
    contours, h = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        print('inside')
        epsilon = 0.01*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(img, [approx], 0, (0), 5)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if len(approx) == 3:
            cv2.putText(img, "Triangle", (x, y), font, 1, (0))
        elif len(approx) == 4:
            cv2.putText(img, "Rectangle", (x, y), font, 1, (0))
        elif len(approx) == 5:
            cv2.putText(img, "Pentagon", (x, y), font, 1, (0))
        elif 6 < len(approx) < 15:
            cv2.putText(img, "Ellipse", (x, y), font, 1, (0))
        else:
            cv2.putText(img, "Circle", (x, y), font, 1, (0))

    
    cv2.imshow("shapes", img)
    cv2.imshow("Threshold", threshold)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    #exit()
    # show the output image
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)

    # show the output image
    #cv2.imshow("resized", resized)
    #cv2.waitKey(0)

    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)
    #cv2.waitKey(0)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #cv2.imshow("blurred", blurred)
    #cv2.waitKey(0)

    #thresh = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)[1]
    #cv2.imshow("thresh", thresh)
    #cv2.waitKey(0)

    # find contours in the thresholded image and initialize the
    # shape detector
    cnts = cv2.findContours(blurred.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    sd = ShapeDetector()

    print(len(cnts))

    # loop over the contours
    for c in cnts:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        shape = sd.detect(c)
    
        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 2)
    
        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, 100)
    lines = cv2.HoughLines(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    
    print(circles)
    print(lines)

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
    
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    
        # show the output image
        cv2.imshow("output", np.hstack([image, output]))
        cv2.waitKey(0)