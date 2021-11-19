import diceRecognition
import os
import cv2 as cv

def getFiles(path, ext=['.jpg', '.png', '.jpeg']):
    files = []
    for f in os.listdir(path):
        if(os.path.isfile(os.path.join(path, f)) and os.path.splitext(f)[1] in ext):
            files.append(f)
    return files

if __name__ == "__main__":
    files = getFiles(os.path.dirname(__file__) + "\\resources\\dices")

    for file in files:
        full, dices = diceRecognition.recognize(file)
        cv.imshow("dice", full)
        cv.imshow("pits", dices)
        cv.waitKey(0)
        cv.destroyAllWindows()