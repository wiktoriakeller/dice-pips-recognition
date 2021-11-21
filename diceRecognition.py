from functools import total_ordering
from typing import Tuple
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

    #check whether image is in grayscale, BGR image
    #will have length of the shape equal to 3
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
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    dices = []
    positions = []
    imageForCropping = np.copy(drawImg)

    for cnt in contours:
        area = cv.contourArea(cnt)

        if area >= 1000:
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.int0(box)

            width = int(rect[1][0])
            height = int(rect[1][1])

            if abs(width - height) <= 40:
                positions.append((box[0][0], box[0][1]))
                cv.drawContours(drawImg, [box], 0, (0, 255, 0), 2)
                sourcePoints = box.astype("float32")
                targetPoints = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32")
                M = cv.getPerspectiveTransform(sourcePoints, targetPoints)
                warped = cv.warpPerspective(imageForCropping, M, (width, height))
                dices.append(warped)

    return dices, positions

def openImage(file):
    path = os.path.dirname(__file__) + "\\resources\\dices"
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
    
def simpleBlobDetection(img, minThreshold, maxThreshold, minArea, maxArea, minCircularity, minInertiaRatio):
    diceImgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
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

    return cv.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
            cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS), len(keypoints)

def applyGamma(img, gamma):
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

   return cv.LUT(img, table) 

def processImage(img, gamma, kernel):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgGammaApplied = applyGamma(imgGray, gamma)

    imgBlur = cv.GaussianBlur(imgGammaApplied, (3, 3), 8)
    imgThreshold = cv.threshold(imgBlur, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    imgMorph = cv.morphologyEx(imgThreshold, cv.MORPH_CLOSE, kernel)
    imgMorph = cv.morphologyEx(imgMorph, cv.MORPH_OPEN, kernel)
    
    imgCanny = cv.Canny(imgMorph, 1, 255)

    return imgCanny

def recognize(fileName):
    #blob detection parameters
    minThreshold = 50                  
    maxThreshold = 200     
    minArea = 60                
    maxArea = 1000
    minCircularity = 0.4
    minInertiaRatio = 0.4

    recognitionResult = open(os.path.dirname(__file__) + "\\results\\simpleRecognition\\" + fileName + ".txt", "w")

    totalPips = 0
    totalDices = 0

    img = openImage(fileName)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    output = processImage(img, 0.5, kernel)

    resultImage = np.copy(img)
    dices, positions = getContours(output, resultImage)
    totalDices = len(dices)

    if totalDices == 0:
        output = processImage(img, 0.2, kernel)
        resultImage = np.copy(img)
        dices, positions = getContours(output, resultImage)
        totalDices = len(dices)

    cv.putText(resultImage, "Number of dices: " + str(totalDices), (30, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    filteredDices = []

    if totalDices > 0:
        for i in range(len(dices)):
            dices[i] = resizeToSize(dices[i], 128)

        imgArea = dices[0].shape[0] * dices[0].shape[1]
        maxArea = int(imgArea / 2)

        for i in range(len(dices)):
            diceMorph = cv.morphologyEx(dices[i], cv.MORPH_CLOSE, kernel)
            diceMorph = cv.morphologyEx(diceMorph, cv.MORPH_OPEN, kernel)

            imgWithKeypoints, number = simpleBlobDetection(diceMorph, minThreshold, maxThreshold, minArea, maxArea, minCircularity, minInertiaRatio)

            if number == 0:
                diceGamma = applyGamma(dices[i], 0.28)
                diceMorph = cv.morphologyEx(diceGamma, cv.MORPH_CLOSE, kernel)
                diceMorph = cv.morphologyEx(diceMorph, cv.MORPH_OPEN, kernel)
                imgWithKeypoints, number = simpleBlobDetection(diceMorph, minThreshold, maxThreshold, minArea, maxArea, minCircularity, minInertiaRatio)

            cv.putText(imgWithKeypoints, str(number), (5, 25), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv.putText(resultImage, str(number), positions[i], cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            filteredDices.append(imgWithKeypoints)
            totalPips += number
            recognitionResult.write("Dice " + str(i) + " pips: " + str(number) + "\n")

        print("File: " + fileName)
        print("Total dices found: " + str(totalDices))
        print("Total pips found: " + str(totalPips))
        print()

        recognitionResult.write("Total dices: " + str(totalDices) + "\n")
        recognitionResult.write("Total pips: " + str(totalPips) + "\n")

        full = joinImages(0.6, [img, resultImage], True)
        dices = joinImages(0.5, filteredDices, True)

        return full, dices

    print("File: " + fileName)
    print("Total dices found: " + "0")
    print("Total pips found: " + "0", end="/n/n")
    print()

    recognitionResult.write("Total dices: 0" + "\n")
    recognitionResult.write("Total pips: 0" + "\n\n")
    
    return resultImage, resultImage
