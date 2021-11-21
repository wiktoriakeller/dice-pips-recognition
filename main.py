import diceRecognition
import os
import cv2 as cv

computed = {}

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
        if i in computed:
            full, dices = computed[i]
        else:
            full, dices = diceRecognition.recognize(files[i])
            computed[i] = (full, dices)

        cv.imshow("dice", full)
        cv.imshow("pits", dices)
        key = cv.waitKey(0)

        if chr(key % 256) == 'q' or chr(key % 256) == 'Q':
            break
        elif chr(key % 256) == 'd' or chr(key % 256) == 'D':
            i = (i + 1) % len(files)
        elif chr(key % 256) == 'a' or chr(key % 256) == 'A':
            i = (i - 1) % len(files)

    cv.destroyAllWindows()
    