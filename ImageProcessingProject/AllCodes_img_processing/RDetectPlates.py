# all imports -----------------------------
import cv2
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import RLpDefinitions as definitions
import RLpPreprocess as preprocess
import RRecognizeChar
from RLicensePlate import RLicensePlate
from RPossibleChar import RPossibleChar as PossibleChar
from imutils import perspective
import imutils


# new imports :
from sklearn.neural_network import MLPClassifier

# ----------------------------------
#          Main Class function
# ---------------------------------
class RDetectPlates:
    def __init__(self, imgOrg, showSteps=True):
        self.imgOriginal = imgOrg
        self.showSteps = showSteps
        self.resY, self.resX, ch = self.imgOriginal.shape

    # getFilteredImageFromRegion #############
    def getFilteredImageFromRegion(self, plate, pltcoord):
        #plate = perspective.four_point_transform(self.imgOriginal, region)
        plate = imutils.resize(plate, width=400)
        # show to decide beste capturing
        #plt.imshow(plate)  # there is a problem for closed images
        #plt.close()
        # draw rectangle around detected plate regions
        #cv2.drawContours(self.imgOriginal, [region], -1, definitions.RGB_BLUE, 5)
        cv2.rectangle(self.imgOriginal, (pltcoord[0], pltcoord[1]), \
                      (pltcoord[0] + pltcoord[2], pltcoord[1] + pltcoord[3]), (200, 0, 0), 2)
        grayplate, processed_plate  = preprocess.kmeans_seg(plate)

        #btw_and = cv2.bitwise_and(blackhat, imgThresh)
        #if self.showSteps:
        #    plt.imshow("1.a blackhat", blackhat)
        #    plt.imshow("1.b imgThresh", imgThresh)
        #    plt.imshow("1.c btw_and", btw_and)
        #   plt.waitKey(0)
        #    plt.close()
        return plate, processed_plate #btw_and, blackhat, imgThresh
    # detectPlates ########
    def detectPlates(self):
        listOfPlates = []                                   # this will be possible plates
        #regions, coords = self.detectPossiblePlateRegions() # this part is for plate detection and its improved
        # my codes
        plates, coords = self.detectPossiblePlateRegions()
        #counter = 0
        # for every possible plate regions
        for counter, plate in enumerate(plates):
            pltCoord = coords[counter]
            #counter += 1
            grayPlate, processed_plate = self.getFilteredImageFromRegion(plate, pltCoord)
            possibleCharsList = self.findPossibleCharsInPlate(grayPlate, processed_plate)
            # my algorithm will not need this  -----   - - -  - - - - - - - - -

            matchingCharList  = self.matchingChar(possibleCharsList)

            if matchingCharList == None: continue
# for showing steps if needed ------------------------------------------
            if self.showSteps:
                height, width = processed_plate.shape
                imgContours   = np.zeros((height, width, 3), np.uint8)
                for matchingChar in matchingCharList:
                    intRandomBlue  = random.randint(0, 255)
                    intRandomGreen = random.randint(0, 255)
                    intRandomRed   = random.randint(0, 255)
                    contours = []
                    # for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                    cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
                    # cv2.imshow("3 matching chars", imgContours)
                    plt.imshow(imgContours)
            #    cv2.waitKey(0)
            #    ----- - - - --  - - - - - - - -  - - - - -
            matchingCharList = self.removeInnerOverlappingChars(matchingCharList)

# Checking if we are in right direction by looking at the length of the possible chars of plate
            if len(matchingCharList) < definitions.MIN_NUMBER_OF_MATCHING_CHARS: continue
            if len(matchingCharList) > definitions.MAX_NUMBER_OF_MATCHING_CHARS: continue
            matchingCharList.sort(key=lambda matchingChar: matchingChar.intCenterX)  # sort chars from left to right
            lp = RLicensePlate()
            lp.coord = pltCoord
            #lp.region = region # no need to region
            lp.imgPlate = plate
            lp.imgFiltered = processed_plate
            # lp.imgFiltered = imgThresh
            lp.charList = matchingCharList
            lp.charTypeList = self.getCharTypes(matchingCharList)
            lp.strChars = RRecognizeChar.recognizeCharsInPlate(lp)
            listOfPlates.append(lp)
        return listOfPlates

    def getCharTypes(self, listOfMatchingChars):

        # sortedRects = sorted(rects, key=self.getKey)
        distList = []; meanDist = 0
        numChars = len(listOfMatchingChars)
        for i in range(numChars-1):
            distList.append(listOfMatchingChars[i+1].intCenterX - listOfMatchingChars[i].intCenterX)
            dist = listOfMatchingChars[i+1].intCenterX - listOfMatchingChars[i].intCenterX
            meanDist += dist
#
        if (numChars-1) != 0:
            meanDist = meanDist / ((numChars - 1))
        else: meanDist = 10

        biggerThanMeanDistCount = 0
        for dist in distList:
            if dist > meanDist+3:
                biggerThanMeanDistCount += 1

        charList = []
        # charList.append("number")
        charList.append("number")
        char = "number"
        flag = "firstNumber"
        digitCount = 1
        maxDigitCount = 2
        for i in range(1,numChars):
            dist = distList[i-1]
            if dist > meanDist+3 and dist < meanDist*3 or digitCount == maxDigitCount:
                if digitCount == maxDigitCount:
                    digitCount = 0
                if flag == "firstNumber":
                    char = "letter"
                    flag = "letter"
                elif flag == "letter":
                    char = "number"
                    flag = "secondNumber"
                if flag == "letter":
                    maxDigitCount = 3
                # if flag == "secondNumber":
                #     maxDigitCount = 4
            digitCount += 1
            charList.append(char)

        newCharList = []
        if biggerThanMeanDistCount > 2:
            newCharList.append("number")
            newCharList.append("number")
            newCharList.append("letter")
            for i in range(3,numChars-2):
                newCharList.append("both")
            newCharList.append("number")
            newCharList.append("number")
            charList = newCharList

        # print("charlist : " + str(charList))
        return charList

## ---------------------------------------------------------------------------------------------------------------------
    # detectPossiblePlateRegions ########
## ---------------------------------------------------------------------------------------------------------------------
    def detectPossiblePlateRegions(self): # this part is done (it is improved)
        ## new codes goes here
        #retRegions = []  # this will be the return value
        gCoords = []  # this will be the return value
        #retCoords = []
        poss_plates=[]
        globCoord=[]

        # # defining kernels
        #
        # Vertical Kernels
        #vertKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
        #pKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        #
        # Horizontal Kernels
        bKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        b2Kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        #smallKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 3))
        #HKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        longKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 4))  # the rectangle kernel
        #superKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))  # 27,3 check 29 also
        #
        #
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))

        # convert the image to grayscale, and apply the blackhat operation
        gimg = cv2.cvtColor(self.imgOriginal, cv2.COLOR_BGR2GRAY)
        Pgray = cv2.morphologyEx(gimg, cv2.MORPH_BLACKHAT, rectKernel)
        # find regions in the image that are light
        Fcolor = cv2.morphologyEx(Pgray, cv2.MORPH_CLOSE, rectKernel)
        Fcolor = cv2.threshold(Fcolor, 0, 255, cv2.THRESH_BINARY)[1]
        # compute the Scharr gradient representation of the blackhat image and scale the
        # resulting image into the range [0, 255]
        gradX = np.absolute(cv2.Sobel(Pgray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1))
        gradX = (255 * ((gradX - gradX.min()) / (gradX.max() - gradX.min()))).astype("uint8")

        # blur the gradient representation, apply a closing operation, and threshold the
        # image using Otsu's method
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        gradX = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #
        gradX = cv2.dilate(cv2.erode(gradX, squareKernel, iterations=2), longKernel, iterations=5)
        #
        Pgray = cv2.bitwise_and(gradX, gradX, mask=Fcolor)
        Pgray = cv2.dilate(cv2.erode(Pgray, squareKernel, iterations=2), squareKernel, iterations=2)
        # find contours in the thresholded image
        _,cnts,_ = cv2.findContours(Pgray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # loop over the contours
        bigpics = []  # this will not return a value
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            aspectRatio = w / float(h)
            # compute the rotated bounding box of the region
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))
            if (aspectRatio > 2 and aspectRatio < 8) and h > 10 and w > 50 and h < 125 and w < 400:
                bigpics.append(box)
                gCoords.append(np.array([x, y, w, h]))

        for ccounter,bigpic in enumerate(bigpics):
            #plt.close()
            recheck = True
            gray = perspective.four_point_transform(gimg, bigpic)
            #plt.imshow(gray)
            #plt.close()
            # color based middle filter design: (No gradient)
            # preprocessing
            # Left filter: gradient based
            gradX = np.absolute(cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1))
            gradY = np.absolute(cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1))
            gradX = (255 * ((gradX - gradX.min()) / (gradX.max() - gradX.min()))).astype("uint8")
            gradX = cv2.medianBlur(cv2.GaussianBlur(gradX, (15, 15), 10), 7)
            gradY = (255 * ((gradY - gradY.min()) / (gradY.max() - gradY.min()))).astype("uint8")
            gradY = cv2.medianBlur(cv2.GaussianBlur(gradY, (5, 5), 0), 5)
            left_gray = (gradX + 2*gradY)//2
            #left_gray = cv2.medianBlur(cv2.GaussianBlur(left_gray, (5, 5), 0), 5)
            # Decision parameters are going here
            left_gray = cv2.threshold(left_gray, .75 * left_gray.max(), left_gray.max(),  cv2.THRESH_TRUNC)[1]
            left_gray = cv2.erode(left_gray, bKernel, iterations=1)
            left_gray = cv2.dilate(left_gray, b2Kernel, iterations=3)
            left_gray = cv2.threshold(left_gray, .8 * left_gray.max(), left_gray.max(),  cv2.THRESH_BINARY)[1]
            left_gray = cv2.Canny(left_gray, 0.90 * left_gray.max(), left_gray.max())
             # todo: should decide about this parameter
            left_gray = cv2.erode(left_gray, b2Kernel, iterations=1)
            left_gray = cv2.dilate(left_gray, b2Kernel, iterations=4)

            #plt.imshow(left_gray)
            #plt.close()

            # # todo: should decide about this parameter
            # finalg = cv2.dilate(finalg, bKernel, iterations=3)
            #
            # plt.imshow(finalg)
            # plt.close()
            cnts = cv2.findContours(left_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
            if len(cnts) == 1:
                for c in cnts:
                    # grab the bounding box associated with the contour and compute the area and
                    # aspect ratio
                    (x, y, w, h) = cv2.boundingRect(c)
                    aspectRatio = w / float(h)
                    # compute the rotated bounding box of the region
                    rect = cv2.minAreaRect(c)
                    box = np.int0(cv2.boxPoints(rect))

                    # ensure the aspect ratio, width, and height of the bounding box fall within
                    # tolerable limits, then update the list of license plate regions
                    # todo: lets examin the scaled factores with some tolerance
                    if (aspectRatio > 2.5 and aspectRatio < 8) and h > 10 and w > 50 and h < 125 and w < 400:
                        platel = perspective.four_point_transform(gray, box)
                        poss_plates.append(platel)
                        #plt.imshow(platel)
                        #plt.close()
                        # global coordinate
                        x += gCoords[ccounter][0]
                        y += gCoords[ccounter][1]
                        w = gCoords[ccounter][2]
                        h = gCoords[ccounter][3]
                        #
                        globCoord.append(np.array([x, y, w, h]))
                        #
                        recimg = cv2.rectangle(self.imgOriginal, (x, y), (x + w, y + h), (200, 0, 0), 2)
                        #plt.imshow(recimg)
                        #plt.close()
            # if len(cnts)>1 and len(cnts) < 15 recheck:
            # todo: the condition  "len(cnts) < 15" is a parameter that should be checked
            elif len(cnts) > 1 and len(cnts) < 200:
                cvxhs = self.convexhull(cnts)
                for cvxh in cvxhs:
                    (x, y, w, h) = cv2.boundingRect(cvxh)
                    aspectRatio = w / float(h)
                    # compute the rotated bounding box of the region
                    rect = cv2.minAreaRect(cvxh)
                    box = np.int0(cv2.boxPoints(rect))
                    # todo: parameters going here
                    if (aspectRatio > 2 and aspectRatio < 7) and h > 10 and w > 50 and h < 125 and w < 400:
                        platel = perspective.four_point_transform(gray, box)
                        poss_plates.append(platel)
                        #plt.imshow(platel)
                        #plt.close()
                        # global coordinate
                        x += gCoords[ccounter][0]
                        y += gCoords[ccounter][1]
                        w = gCoords[ccounter][2]
                        h = gCoords[ccounter][3]
                        #
                        globCoord.append(np.array([x, y, w, h]))
                        #recimg = cv2.rectangle(self.imgOriginal, (x, y), (x + w, y + h), (200, 0, 0), 2)
                        #plt.imshow(recimg)
                        #plt.close()
        return poss_plates, globCoord

## ---------------------------------------------------------------------------------------------------------------------
    # findPossibleCharsInPlate ###############
## ---------------------------------------------------------------------------------------------------------------------
    def findPossibleCharsInPlate(self, grayplt, plate):
        possibleCharsList = []  # this will be the return value

        # # segmentation  --------------------------------------------
        #
        #
        # # find all contours in plate
        # #contours = cv2.findContours(plate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        # _,cnts,_ = cv2.findContours(plate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # plate.copy()
        # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]
        # character = {}
        # # Precomputation
        # for c in cnts:
        #     (x, y, w, h) = cv2.boundingRect(c)
        #     #rect = cv2.minAreaRect(c)
        #     #box = np.int0(cv2.boxPoints(rect))
        #     #
        #     M = cv2.moments(c)
        #
        #     cX = int(M["m10"] / M["m00"])
        #     #cY = int(M["m01"] / M["m00"])
        #
        #     #
        #     #
        #     if h > definitions.MIN_PIXEL_HEIGHT and w > definitions.MIN_PIXEL_WIDTH:
        #         xlb = np.array([0, x - 3]).max(); xrb = np.array([grayplt.shape[1], x + w + 3]).min()
        #         ylb = np.array([0, y - 3]).max(); yrb = np.array([grayplt.shape[0], y + h + 3]).min()
        #         cl_img = grayplt[ylb: yrb, xlb: xrb]
        #         tr_img = plate[ylb: yrb, xlb: xrb]
        #         #platel = perspective.four_point_transform(grayplt, box)
        #
        #         final1 = cv2.dilate(tr_img, None, iterations=1).shape(-1)
        #         m,n = final1.shape
        #
        #         cv2.drawContours(tr_img, [c], -1, (0, 255, 0), 2)
        #         #cv2.circle(cl_img, (cX, cY), 7, (255, 255, 255), -1)
        #
        #         plt.imshow(cl_img)
        #         plt.imshow(tr_img)
        #
        #         mlp = MLPClassifier(hidden_layer_sizes=(20, 3), max_iter=150, alpha=1e-4,
        #                             solver='sgd', verbose=10, tol=1e-4, random_state=1,
        #                             learning_rate_init=.1)
        #         prediction = mlp.predict(final1)
        #
        #         # pre pros mke shape
        #
        #         # if prediction < 130:      # update upon valid values
        #         character.update({prediction: cX})
        #
        #        # sort the dictionary
        #        SortedChar = sorted(character, key = character.__getitem__)


        # Assuming that I have data :
        # 0. getting the char
        # 1. feed char to the trained network
        # 2. get networks results:
        # 3. order the results based on the contour center:
        # outpu the   complete prediction

        ## Fatih's code -----------------------------------------------------
        contours = cv2.findContours(plate.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
        # if showSteps is true we will need empty image
        height, width = plate.shape
        imgContours = np.zeros((height, width, 3), np.uint8)

        for i in range(0, len(contours)):
            possibleChar = PossibleChar(contours[i])
            # if contour is a possible char, note this does not compare to other chars (yet) . . .
            if self.checkIfPossibleChar(possibleChar):
                # and add to list of possible chars
                possibleCharsList.append(possibleChar)
                if self.showSteps == True: cv2.drawContours(imgContours, contours, i, definitions.RGB_WHITE)

        if self.showSteps == True: plt.imshow(imgContours)
        #     #cv2.imshow("2 possible chars", imgContours)
        #     # cv2.waitKey(0)
        return possibleCharsList

    # checkIfPossibleChar ######################
    def checkIfPossibleChar(self, possibleChar):
        # this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
        # note that we are not (yet) comparing the char to other chars to look for a group
        if (possibleChar.intBoundingRectArea > definitions.MIN_PIXEL_AREA and
            possibleChar.intBoundingRectWidth > definitions.MIN_PIXEL_WIDTH and
            possibleChar.intBoundingRectHeight > definitions.MIN_PIXEL_HEIGHT and
            possibleChar.fltAspectRatio > definitions.MIN_ASPECT_RATIO and
            possibleChar.fltAspectRatio < definitions.MAX_ASPECT_RATIO):
            return True
        else:
            return False

    def matchingChar(self,listOfPossibleChars):
        if len(listOfPossibleChars) < definitions.MIN_NUMBER_OF_MATCHING_CHARS: return None
            # the purpose of this function is, given a possible char and a big list of possible chars,
            # find all chars in the big list that are a match for the single possible char, and return those matching chars as a list
        listOfMatchingChars = []  # this will be the return value
        meanH = 0; meanY = 0; meanMidY = 0;

        for count, possibleChar in enumerate(listOfPossibleChars):
            meanH    += possibleChar.intBoundingRectHeight
            meanY    += possibleChar.intBoundingRectY
            meanMidY += possibleChar.intCenterY
        meanY    /= count
        meanH    /= count
        meanMidY /= count
        count = 0
        for possibleChar in listOfPossibleChars:
            for possibleMatchingChar in listOfPossibleChars:  # for each char in big list
                # possibleMatchingChar = listOfPossibleChars[i]
                if possibleChar == possibleMatchingChar: continue
                    # if the char we attempting to find matches for is the exact same char as the char in the big list we are currently checking
                    # then we should not include it in the list of matches b/c that would end up double including the current char
                      # so do not add to list of matches and jump back to top of for loop
                    # end if     # compute stuff to see if chars are a match   # thank you Fatih with described stuff :)
                fltDistanceBetweenChars = self.distanceBetweenChars(possibleChar, possibleMatchingChar)

                fltAngleBetweenChars    = self.angleBetweenChars(possibleChar, possibleMatchingChar)

                fltChangeInArea = float(
                    abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(
                    possibleChar.intBoundingRectArea)

                fltChangeInWidth = float(
                    abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(
                    possibleChar.intBoundingRectWidth)
                fltChangeInHeight = float(
                    abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(
                    possibleChar.intBoundingRectHeight)

                # check if chars match
                if (fltDistanceBetweenChars < (
                    possibleChar.fltDiagonalSize * definitions.MAX_DIAG_SIZE_MULTIPLE_AWAY) and
                            fltAngleBetweenChars < definitions.MAX_ANGLE_BETWEEN_CHARS and
                            fltChangeInArea < definitions.MAX_CHANGE_IN_AREA and
                            fltChangeInWidth < definitions.MAX_CHANGE_IN_WIDTH and
                            fltChangeInHeight < definitions.MAX_CHANGE_IN_HEIGHT):

                    res = 15
                    if possibleChar.intBoundingRectY > meanY-res and \
                            possibleChar.intBoundingRectY < meanY + res and \
                            possibleChar.intBoundingRectHeight > meanH-res and \
                            possibleChar.intBoundingRectHeight < meanH+res and \
                            possibleChar.intCenterY > meanMidY-res and \
                            possibleChar.intCenterY < meanMidY+res and \
                            possibleChar.intBoundingRectY+possibleChar.intBoundingRectHeight > meanY+meanH-res and \
                            possibleChar.intBoundingRectY+possibleChar.intBoundingRectHeight < meanY+meanH+res:
                    # res = 7
                    # if possibleChar.intBoundingRectY > meanY - res and \
                    #                 possibleChar.intBoundingRectY < meanY + res and \
                    #                 possibleChar.intBoundingRectY > meanY - res and \
                    #                 possibleChar.intBoundingRectHeight > meanH - res and \
                    #                 possibleChar.intBoundingRectHeight < meanH + res and \
                    #                 possibleChar.intCenterY > meanMidY - res and \
                    #                 possibleChar.intCenterY < meanMidY + res and \
                    #                         possibleChar.intBoundingRectY + possibleChar.intBoundingRectHeight > meanY + meanH - res and \
                    #                         possibleChar.intBoundingRectY + possibleChar.intBoundingRectHeight < meanY + meanH + res:
                        if count != 0:
                            if listOfMatchingChars[count-1] != possibleChar:
                                listOfMatchingChars.append(possibleChar)
                                count+=1
                        else:
                            listOfMatchingChars.append(possibleChar)
                            count+=1

        return listOfMatchingChars  # return result

    # use Pythagorean theorem to calculate distance between two chars
    def distanceBetweenChars(self, firstChar, secondChar):
        intX = abs(firstChar.intCenterX - secondChar.intCenterX)
        intY = abs(firstChar.intCenterY - secondChar.intCenterY)
        return math.sqrt((intX ** 2) + (intY ** 2))

    # use basic trigonometry (SOH CAH TOA) to calculate angle between chars
    def angleBetweenChars(self, firstChar, secondChar):
        fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
        fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

# angle as a feature?  lets check if it is a good idea

        if fltAdj != 0.0:  # check to make sure we do not divide by zero if the center X positions are equal, float division by zero will cause a crash in Python
            fltAngleInRad = math.atan(fltOpp / fltAdj)  # if adjacent is not zero, calculate angle
        else:
            fltAngleInRad = 1.5708  # if adjacent is zero, use this as the angle, this is to be consistent with the C++ version of this program

        fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)  # calculate angle in degrees

        return fltAngleInDeg

    def removeInnerOverlappingChars(self, listOfMatchingChars):
        listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)                # this will be the return value
        for currentChar in listOfMatchingChars:
            for otherChar in listOfMatchingChars:
                if currentChar != otherChar:        # if current char and other char are not the same char . . .
                                                                                # if current char and other char have center points at almost the same location . . .
                    if self.distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * definitions.MIN_DIAG_SIZE_MULTIPLE_AWAY):
                        # if we get in here we have found overlapping chars
                        # next we identify which char is smaller, then if that char was not already removed on a previous pass, remove it
                        if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:         # if current char is smaller than other char
                            if currentChar in listOfMatchingCharsWithInnerCharRemoved:              # if current char was not already removed on a previous pass . . .
                                listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)         # then remove current char
                            # end if
                        else:                                                                       # else if other char is smaller than current char
                            if otherChar in listOfMatchingCharsWithInnerCharRemoved:                # if other char was not already removed on a previous pass . . .
                                listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)           # then remove other char

        return listOfMatchingCharsWithInnerCharRemoved


    # convexhull functions  --- --- - - -- -- --  - -- - -
    def find_if_close(self, cnt1, cnt2):
            row1, row2 = cnt1.shape[0], cnt2.shape[0]
            for i in range(row1):
                for j in range(row2):
                    dist = np.linalg.norm(cnt1[i] - cnt2[j])
                    # todo: decision parameter goes here
                    if abs(dist) < 60:
                        return True
                    elif i == row1 - 1 and j == row2 - 1:
                        return False

        # convexhull drawıng system
    def convexhull(self, contours):
            LENGTH = len(contours)
            status = np.zeros((LENGTH, 1))

            for i, cnt1 in enumerate(contours):
                x = i
                if i != LENGTH - 1:
                    for j, cnt2 in enumerate(contours[i + 1:]):
                        x = x + 1
                        dist = self.find_if_close(cnt1, cnt2)
                        if dist == True:
                            val = min(status[i], status[x])
                            status[x] = status[i] = val
                        else:
                            if status[x] == status[i]:
                                status[x] = i + 1

            unified = []
            maximum = int(status.max()) + 1
            for i in range(maximum):
                pos = np.where(status == i)[0]
                if pos.size != 0:
                    cont = np.vstack(contours[i] for i in pos)
                    hull = cv2.convexHull(cont)
                    unified.append(hull)

            return unified



####################################################################################
##                             This part can be escaped
####################################################################################
    # def extractPlate(self,imgOriginal, listOfMatchingChars):
    #     possiblePlate = PossiblePlate()  # this will be the return value
    #
    #     listOfMatchingChars.sort(
    #         key=lambda matchingChar: matchingChar.intCenterX)  # sort chars from left to right based on x position
    #
    #     # calculate the center point of the plate
    #     fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[
    #         len(listOfMatchingChars) - 1].intCenterX) / 2.0
    #     fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[
    #         len(listOfMatchingChars) - 1].intCenterY) / 2.0
    #
    #     ptPlateCenter = fltPlateCenterX, fltPlateCenterY
    #
    #     # calculate plate width and height
    #     intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[
    #         len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[
    #                              0].intBoundingRectX) * definitions.PLATE_WIDTH_PADDING_FACTOR)
    #
    #     intTotalOfCharHeights = 0
    #
    #     for matchingChar in listOfMatchingChars:
    #         intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
    #     # end for
    #
    #     fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)
    #
    #     intPlateHeight = int(fltAverageCharHeight * definitions.PLATE_HEIGHT_PADDING_FACTOR)
    #
    #     # calculate correction angle of plate region
    #     fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    #     fltHypotenuse = self.distanceBetweenChars(listOfMatchingChars[0],
    #                                                      listOfMatchingChars[len(listOfMatchingChars) - 1])
    #     fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    #     fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)
    #
    #     # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    #     possiblePlate.rrLocationOfPlateInScene = (
    #     tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg)
    #
    #     # final steps are to perform the actual rotation
    #
    #     # get the rotation matrix for our calculated correction angle
    #     # rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)
    #
    #     height, width, numChannels = imgOriginal.shape  # unpack original image width and height
    #
    #     # imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))  # rotate the entire image
    #     #
    #     # imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))
    #
    #     possiblePlate.imgPlate = imgOriginal  # copy the cropped plate image into the applicable member variable of the possible plate
    #
    #     return possiblePlate