import cv2 as cv
import numpy as np
import os 

def resizeToSize(img, size):
    return cv.resize(img, (size, size))

def resizeToImage(img1, img2):
    return cv.resize(img1, (img2.shape[1], img2.shape[0]), None, 1, 1)

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

def openImage(file):
    path = os.path.dirname(__file__) + "\\resources\\dices\\"
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
        print("Can't open file: " + path)
        exit(1)

    return img

def findContoursInDice(img):
    diceImgGray = cv.cvtColor(dices[i], cv.COLOR_BGR2GRAY)

    diceImgBlur = cv.GaussianBlur(diceImgGray, (3, 3), 50, 50)
    diceImgThreshold = cv.threshold(diceImgBlur, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    diceImgCanny = cv.Canny(diceImgThreshold, 20, 255)
    contours, hierarchy = cv.findContours(diceImgCanny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    return contours
    
def simpleBlobDetection(img, minThreshold, maxThreshold, minArea, maxArea, minCircularity, minInertiaRatio):
    diceImgGray = cv.cvtColor(dices[i], cv.COLOR_BGR2GRAY)
    
    params = cv.SimpleBlobDetector_Params()  
    params.filterByArea = True
    params.filterByCircularity = True
    params.filterByInertia = True
    params.minThreshold = minThreshold
    params.maxThreshold = maxThreshold
    params.minArea = minArea
    params.maxArea = maxArea
    params.minCircularity = minCircularity
    params.minInertiaRatio = minInertiaRatio
    detector = cv.SimpleBlobDetector_create(params)

    keypoints = detector.detect(diceImgGray)

    #invImage = cv.bitwise_not(diceImgGray)
    #keypoints2 = detector.detect(invImage) + keypoints
    return cv.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS), len(keypoints)

def deleteGlares(img):
    pass

#blob detection parameters
minThreshold = 50                  
maxThreshold = 200     
minArea = 60                
maxArea = 1000
minCircularity = 0.4
minInertiaRatio = 0.4

img = openImage("dice1.jpg")
blank = np.zeros_like(img)

imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgBlur = cv.GaussianBlur(imgGray, (3, 3), 10)

imgThreshold = cv.threshold(imgBlur, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
imgThreshold = cv.morphologyEx(imgThreshold, cv.MORPH_CLOSE, kernel)
imgThreshold = cv.morphologyEx(imgThreshold, cv.MORPH_OPEN, kernel)

imgCanny = cv.Canny(imgThreshold, 1, 255)
resultImage = np.copy(img)

dices = getContours(imgCanny, resultImage)

cv.putText(resultImage, "Number of dices: " + str(len(dices)), (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
filteredDices = []

for i in range(len(dices)):
    dices[i] = resizeToSize(dices[i], 128)
    
imgArea = dices[0].shape[0] * dices[0].shape[1]
maxArea = int(imgArea / 2)

for i in range(len(dices)):
    blur = cv.medianBlur(dices[i], 5)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    lowerGray = np.array([0, 5, 50], np.uint8)
    upperGray = np.array([179, 50, 255], np.uint8)
    maskGray = cv.inRange(hsv, lowerGray, upperGray)
    mean = np.mean(maskGray)

    for j in range(maskGray.shape[0]):
        for g in range(maskGray.shape[1]):
            if(maskGray[j][g] >= 220 and mean <= 30):
                dices[i][j][g][0] = 0
                dices[i][j][g][1] = 0
                dices[i][j][g][2] = 0

    imgWithKeypoints, number = simpleBlobDetection(dices[i], minThreshold, maxThreshold, minArea, maxArea, minCircularity, minInertiaRatio)
    cv.putText(imgWithKeypoints, str(number), (5, 25), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    filteredDices.append(imgWithKeypoints)

joined = joinImages(0.4, [[img, imgBlur, imgThreshold], [imgCanny, resultImage, blank]], False)
cv.imshow("images", joined)

joined = joinImages(0.5, filteredDices, True)
cv.imshow("pits", joined)

cv.waitKey(0)
cv.destroyAllWindows()