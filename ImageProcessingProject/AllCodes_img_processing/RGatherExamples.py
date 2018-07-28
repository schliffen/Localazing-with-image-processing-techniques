import cv2
import numpy as np
import os
import random

import RLpDefinitions as definitions
import RDetectPlates as detectPlates

from imutils import paths


savePath = "set"
counts = {}

showSteps = False

def gatherExamples(imgOriginal):
    global path

    lpDetector = detectPlates.RDetectPlates(imgOriginal, showSteps=showSteps)
    plates = lpDetector.detectPlates()

    if len(plates) < 1:
        # temp = cv2.resize(imgOriginal, (720, 405))
        temp = cv2.resize(imgOriginal, (1280, 720))
        cv2.imshow("imgOriginal", temp)
        cv2.waitKey(1)
        return
    count = 0
    for lp in plates:
        count += 1
        region = lp.region
        plate = lp.imgPlate
        coord = lp.coord
        filteredPlate = lp.imgFiltered
        matchingCharList = lp.charList
        strPlate = lp.strChars
        print("plate: " + strPlate)

        cv2.drawContours(imgOriginal, [region], -1, definitions.RGB_YELLOW, 2)
        cv2.putText(imgOriginal, strPlate, (int(coord[0] - (coord[0] / 8)), int(coord[1] - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, definitions.RGB_RED, 4)

        # temp = cv2.resize(imgOriginal, (720, 405))
        temp = cv2.resize(imgOriginal, (1280, 720))

        cv2.imshow("imgOriginal", temp)
        cv2.imshow("originalPlate",plate)
        cv2.imshow("filteredPlate",filteredPlate)

        height, width, d = plate.shape
        imgContours = np.zeros((height, width, 3), np.uint8)
        for matchingChar in matchingCharList:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            contours = []

            # for matchingChar in listOfMatchingChars:
            contours.append(matchingChar.contour)
            cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))

            cv2.rectangle(imgContours,
                          (matchingChar.intBoundingRectX, matchingChar.intBoundingRectY),
                          (matchingChar.intBoundingRectWidth+matchingChar.intBoundingRectX, matchingChar.intBoundingRectHeight+matchingChar.intBoundingRectY),
                          (intRandomBlue, intRandomGreen, intRandomRed), 2)

            char = filteredPlate[
                   matchingChar.intBoundingRectY: matchingChar.intBoundingRectY + matchingChar.intBoundingRectHeight,
                   matchingChar.intBoundingRectX: matchingChar.intBoundingRectX + matchingChar.intBoundingRectWidth]

            # temp = cv2.resize(imgContours, (720, 360))
            cv2.imshow("plate", imgContours)
            cv2.imshow("char", char)

            key = cv2.waitKey(0) & 0xFF

            if key == ord("`"):
                print("[IGNORING] {}".format(path))
                continue

            key = chr(key).upper()
            dirPath = "{}/{}".format(savePath, key)

            # if the output directory does not exist, create it
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)

            # write the labeled character to file
            count = len(list(paths.list_images(dirPath))) + 1
            path = "{}/{}.png".format(dirPath, str(count).zfill(5))
            cv2.imwrite(path, char)

            # increment the count for the current key
            counts[key] = count + 1

    # key = cv2.waitKey(50) & 0xFF
    # # cv2.destroyAllWindows()
    # if key == ord("e") or key == 27:
    #     exit(0)


counter = 0
import datetime

# dataset = "/media/Depo/Datasets/Securidar/trial/trafidarZoom/20170713/11_32_26.h264"

from os import walk

# datasetPath="/media/Depo/Datasets/Trafidar/15052017/VideoRecord/20_02_38/"
datasetPath="/media/Depo/Dataset/14_00_31_1/"
datasetList = []
for (dirpath, dirnames, filenames) in walk(datasetPath):
    datasetList.extend(filenames)
    break
print(str(datasetList))

for dataset in datasetList:
    imgOriginal = cv2.imread(datasetPath+dataset)

    print(datasetPath+dataset)




    gatherExamples(imgOriginal)
