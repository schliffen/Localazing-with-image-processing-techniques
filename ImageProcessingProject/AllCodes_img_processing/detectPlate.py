import cv2
import random
import datetime
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt
# importig python files-----------------------
import RLpDefinitions as definitions
import RDetectPlates as detectPlates

showSteps = False
#showSteps = True

# wait for every image or not
stepFlag = False

# show or hide blue plate rects.
definitions.showBlueRect = False
# skip image count
counterStep = 1
# imageCounter = 0

drFlag = None

def lpDetect(image):
    global drFlag, imageCounter

    resY, resX, ch = image.shape

    #   ---------   starting by step -01 -> the plate detection improvements
    # detect plates, do segmentation and recognize charaters
    # class of the detector
    #
    lpDetector = detectPlates.RDetectPlates(image, showSteps = showSteps)
    plates     = lpDetector.detectPlates()
    #
    found = False
    for count,lp in enumerate(plates):

        #region = lp.region
        plate = lp.imgPlate
        coord = lp.coord
        matchingCharList = lp.charList
        strPlate = lp.strChars
        filteredPlate = lp.imgFiltered

        # cv2.imwrite("plates/"+str(imageCounter)+".png", plate)
        # imageCounter+=1

        print("plate: " + strPlate)

        # cv2 computations goes here -----
        #cv2.drawContours(image, [region], -1, definitions.RGB_YELLOW, 5)
        cv2.rectangle(image, (coord[0], coord[1]), \
                      (coord[0] + coord[2], coord[1] + coord[3]), (0, 200, 0), 3)

        cv2.putText(image, strPlate, (int(coord[0] - (coord[0] / 8)), int(coord[1] - 30)), \
                    cv2.FONT_HERSHEY_SIMPLEX, 2.1,(100, 200, 0), 4)

        plt.imshow(image)

        height, width = plate.shape
        imgContours = np.zeros((height, width, 3), np.uint8)

#  --- drawing contours  -------

        for matchingChar in matchingCharList:
            # in order to draw contours of each character in different colors (for clear separation)
            intRandomBlue  = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed   = random.randint(0, 255)
            contours = []

            # for matchingChar in listOfMatchingChars:
            # computing contours of the characters
            contours.append(matchingChar.contour)   # the contour parameter of the char
            cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))

            # cv2.imshow("plate " +str(count), plate)
            # cv2.imshow("matchingChars " +str(count), imgContours)
            # cv2.imshow("filteredPlate " +str(count), filteredPlate)
            plt.imshow(plate)
            plt.imshow(imgContours)
            plt.imshow(filteredPlate)

        found = True

    image = cv2.resize(image, (int(resX/3*2), int(resY/3*2)))
    plt.imshow(image)
    plt.show()
    plt.close()
    key = 0


#  -------------------     what does this part do?
#  masking last characters
    #if found:
    #    key = cv2.waitKey(0)  & 0xFF
    #else:
    #    key = cv2.waitKey(10) & 0xFF
#
    # cv2.destroyAllWindows()
    #if key == ord("e") or key == 27:
    #    exit(0)
# ------------------------------------------

# ------------------------------------------ Videos
# path="/media/Depo/Dataset/16_06_45_1/"
# path="/media/Depo/Dataset/14_05_19_1/"
# path="/media/Depo/Dataset/14_00_31_1/"

# --------------------------------------------------------------Images
# path = "/media/Depo/Radarsan/forANPR/video_16_00_41.h264"
# path = "/media/Depo/Radarsan/sunnyANPR/20_31_52.h264"
# path = "/media/Depo/Radarsan/sunnyANPR/20_29_18.h264"
# path = "/media/Depo/Radarsan/sunnyANPR/20_11_57.h264"
# path = "/media/Depo/Radarsan/sunnyANPR/20_12_20.h264"
#path = "/home/bayes/Academic/Research/Radarsan-01/ANPR/datasets/sunny/20_20_53.h264"
#path = '/media/radarsan/Depo/Datasets/Trafidar/köprü/2017_11_24/16_51_57/'
#path = '/media/radarsan/Depo/Datasets/Trafidar/köprü/ortalamaHız/16_25_43_1'
#path = '/media/radarsan/Depo/Datasets/Trafidar/köprü/2017_11_30/16_41_24'
#path = '/home/bayes/Academic/Research/Radarsan-01/ANPR/DATA/DesignData'
path = '/home/bayes/Academic/Research/Radarsan-01/ANPR/DATA/AverageVelocity/Bridge_1'
setType = "image"
#setType = "video"

if __name__ == "__main__":
    if setType == "image":
        # the path of the images
        imgs = sorted(list(paths.list_images(path)), reverse=True) # lets select in random way
# for main code
        for path in imgs:
            plt.close()
#           print(path)
#           imgPath = path
            #rnd = np.random.randint(0, len(imgs)-1, 1)[0]
            #imgPath = imgs[rnd]
            start = datetime.datetime.now()

            imgOriginal = cv2.imread(path)
            plt.imshow(imgOriginal)
            plt.close()

            lpDetect(imgOriginal)

            end = datetime.datetime.now()
            print("time: " + str(end-start))

    elif setType == "video":
        cap = cv2.VideoCapture(path)

        # skip first 15 image
        for i in range(15):
            cap.read()

        counter = 0
        while True:
            ret, imgOriginal = cap.read()

            if ret == False:
                break

            # process every one of 5 frame
            counter += 1
            if counter % 5 != 0:
                continue

            # crop roi of image
            imgOriginal = imgOriginal[200:1080, 0:1920]

            start = datetime.datetime.now()
            lpDetect(imgOriginal)

            end = datetime.datetime.now()
            print("time: " + str(end-start))