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

    i = 0
    while True:
        full, dices = diceRecognition.recognize(files[i])
        cv.imshow(files[i], full)
        cv.imshow("pits", dices)
        key = cv.waitKey(0)
        if chr(key%256) == 'q':
            break
        elif chr(key%256) == 'a':
            i -= 1
        else:
            i += 1
        cv.destroyAllWindows()