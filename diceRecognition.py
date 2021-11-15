import cv2 as cv
import numpy as np
import os

def resizeToImage(img1, img2):
    image = cv.resize(img1, (img2.shape[1], img2.shape[0]), None, 1, 1)
    return image

def resize(sample, image, scale):
    #check whether sample image and target image have the same dimensions
    if image.shape[:2] == sample.shape[:2]:
        image = cv.resize(image, (0, 0), None, scale, scale)
    else:
        image = cv.resize(image, (sample.shape[1], sample.shape[0]), None, scale, scale)

    #check whether image is in grayscale, BGR image will have length of the shape equal to 3
    if len(image.shape) == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    
    return image

def joinImages(scale, images, oneDimArr = True):
    result = None

    if not(oneDimArr):
        rows = len(images)

        for i in range(rows):
            for j in range(len(images[i])):
                images[i][j] = resize(images[0][0], images[i][j], scale)
    
        blank = np.zeros_like(images[0][0])
        horizontal = [blank] * rows

        for i in range(rows):
            horizontal[i] = np.hstack(images[i])
        result = np.vstack(horizontal)
    else:
        for i in range(len(images)):
            images[i] = resize(images[0], images[i], scale)
        
        result = np.hstack(images)

    return result

def getContours(img, drawImg):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    dices = []
    imageForCropping = np.copy(drawImg)

    for cnt in contours:
        area = cv.contourArea(cnt)

        if area > 1000:
            #alternative method
            #cv.drawContours(drawImg, cnt, -1, (0, 255, 0), 3)
            #peri = cv.arcLength(cnt, True)
            #approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            #x, y, w, h = cv.boundingRect(approx)
            #cv.rectangle(drawImg, (x, y), (x + w, y + h), (255, 0, 0), 2)

            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(drawImg, [box], 0, (0, 255, 0), 2)

            width = int(rect[1][0])
            height = int(rect[1][1])

            sourcePoints = box.astype("float32")
            targetPoints = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32")
            M = cv.getPerspectiveTransform(sourcePoints, targetPoints)
            warped = cv.warpPerspective(imageForCropping, M, (width, height))
            dices.append(warped)

    return dices
            
def empty(arg):
    pass

def openImage(file, path=".\\resources\\dices\\"):
    path = os.path.normpath(path)
    if(not os.path.isdir(path)):
        print("no directory: " + path)
        exit(1)

    path = os.path.join(path, file)
    if(not os.path.isfile(path)):
        print("no file: " + file + "\nat: " + path)
        exit(1)

    try:
        img = cv.imread(path)
    except:
        print("can't open file: " + path)
        exit(1)
    return img

min_threshold = 1
max_threshold = 200
min_area = 1
min_circularity = 1
min_inertia_ratio = 1

directory = os.path.dirname(__file__) + "\\resources\\dices\\"

windowTrackBars = "TrackBars"
cv.namedWindow(windowTrackBars)
cv.resizeWindow(windowTrackBars, 640, 240)

cv.createTrackbar("Gauss sigma x", windowTrackBars, 10, 50, empty)
cv.createTrackbar("Thresh min", windowTrackBars, 149, 255, empty)
cv.createTrackbar("Thresh max", windowTrackBars, 255, 255, empty)
cv.createTrackbar("Canny thresh1", windowTrackBars, 1, 255, empty)
cv.createTrackbar("Canny thresh2", windowTrackBars, 255, 255, empty)

while True:
    img = openImage('dice3.jpg')
    blank = np.zeros_like(img)

    gaussSigmaX = cv.getTrackbarPos("Gauss sigma x", windowTrackBars)
    threshMin = cv.getTrackbarPos("Thresh min", windowTrackBars)
    threshMax = cv.getTrackbarPos("Thresh max", windowTrackBars)
    cannyThresh1 = cv.getTrackbarPos("Canny thresh1", windowTrackBars)
    cannyThresh2 = cv.getTrackbarPos("Canny thresh2", windowTrackBars)

    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray, (5, 5), gaussSigmaX)
    threshold = cv.threshold(imgGray, threshMin, threshMax, cv.THRESH_BINARY)
    imgThreshold = threshold[1]
    imgCanny = cv.Canny(imgThreshold, cannyThresh1, cannyThresh2)
    closing = cv.morphologyEx(imgCanny, cv.MORPH_CLOSE, (5, 5), iterations=5)
    closing = cv.dilate(imgCanny, (5,5), iterations=4)
    closing = cv.erode(closing, (5,5), iterations=2)

    resultImage = np.copy(img)
    dices = getContours(imgCanny, resultImage)

    cv.putText(resultImage, "Number of dices: " + str(len(dices)), (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    newImages = []

    for i in range(len(dices)):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        dices[i] = cv.filter2D(dices[i], -1, kernel)

        params = cv.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.filterByCircularity = True
        params.filterByInertia = True
        params.minThreshold = min_threshold
        params.maxThreshold = max_threshold
        params.minArea = min_area
        params.minCircularity = min_circularity
        params.minInertiaRatio = min_inertia_ratio
    
        detector = cv.SimpleBlobDetector_create(params)
        keypoints = detector.detect(dices[i])
        im_with_keypoints = cv.drawKeypoints(dices[i], keypoints, np.array([]), (0, 0, 255),
                                            cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        newImages.append(im_with_keypoints)

    joined = joinImages(0.35, [[img, imgGray, imgBlur], [imgThreshold, imgCanny, resultImage]], False)
    cv.imshow("images", joined)

    #joined = joinImages(0.8, newImages, True)
    #cv.imshow("images", joined)

    cv.waitKey(1)