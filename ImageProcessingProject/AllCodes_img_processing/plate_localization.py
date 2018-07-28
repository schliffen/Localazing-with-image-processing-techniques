#
# this code is for plate localization
#

# import tensorflow as tf
import cv2
import imutils
import numpy as np
from imutils import paths
import RDetectPlates as detplt
from imutils import perspective
import matplotlib.pyplot as plt

import RLpPreprocess as prpr

#
# trying different images by addressing different paths
#
# path = ''
path = '/path to image data'
# path = '/home/bayes/Academic/Research/Radarsan-01/ANPR/I_data/14_00_30_2'
imgs = sorted(list(paths.list_images(path)), reverse=True)
rnd = np.random.randint(0, len(imgs) - 1, 1)[0]
# rnd = 5 #87
# testing the detector:
det_1 = 'fatih'
det_2 = 'comp'

run = True
# the goal of this part is to find plate!!
while (run):
    run = False
    # for first path
    # imgOrg = cv2.imread(path)   #imgOrg = np.asarray(img)

    # for second path
    imgOrg = cv2.imread(path + '/ANPRCAMERA_701_143.52.jpg')  # imgs[rnd]
    # prpr.preprocess(img)
    # s_x, s_y, ch = img.shape
    # intface = paths.list_images(path) # list()
    # imgOrg = sorted(intface, reverse=True)

    # plt.imshow(imgOrg)
    # plt.close()
    try:
        gimg = cv2.cvtColor(imgOrg, cv2.COLOR_BGR2GRAY)
    except:
        print('there is an error in making img gray')
    #
    # working on Fatihs
    # this part should be checked once again
    #

    detector = 'fatih'
    if detector == det_1:

        retRegions = []  # this will be the return value
        retCoords = []  # this will be the return value
        # defining kernels
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 4))  # the rectangle kernel
        superKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))  # 27,3 check 29 also
        smallKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 3))
        pKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 2))

        # convert the image to grayscale, and apply the blackhat operation
        # gray = cv2.cvtColor(imgOrg, cv2.COLOR_BGR2GRAY)
        #  step one
        gradX = np.absolute(cv2.Sobel(gimg, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1))
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
        # I used one name for readability and memory usage
        # step two
        gray = cv2.medianBlur(cv2.blur(cv2.GaussianBlur(gradX, (9, 5), 0), (9, 5)), 5)

        # dilation should go here
        # step three
        gray = cv2.dilate(gray, rectKernel, iterations=2)

        # testing to put erosion here -> goes bad
        # step four
        gray = cv2.morphologyEx(gray, cv2.MORPH_RECT, superKernel)  # using morph consequently is helpfull
        gray = cv2.morphologyEx(gray, cv2.MORPH_RECT, rectKernel)
        gray = cv2.morphologyEx(gray, cv2.MORPH_RECT, smallKernel)

        # step five
        gray = cv2.dilate(gray, rectKernel, iterations=2)

        # testing erosion here
        # steps six and seven
        gray = cv2.erode(gray, rectKernel, iterations=2)
        gray = cv2.morphologyEx(gray, cv2.MORPH_RECT, rectKernel)
        gray = cv2.morphologyEx(gray, cv2.MORPH_RECT, superKernel)
        # step eight
        gray = cv2.dilate(gray, rectKernel, iterations=2)
        gray = cv2.dilate(gray, superKernel, iterations=1)

        # gray = (gradX + 2 * gradY) // 3
        # gray = cv2.dilate(gray, smallKernel, iterations=15)
        # deleting variables
        # del gradY, gradX
        # step nine
        # this part is going to develop by me
        # Make a list for all possible licence plates
        # todo: My previous code:
        # mx_v = np.amax(gray)
        # _, gray = cv2.threshold(gray, 0.60 * mx_v, mx_v, cv2.THRESH_BINARY)
        # _, cnts, _ = cv2.findContours(gray,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # before: cv2.RETR_EXTERNAL
        # del gray, mx_v

        poss_plate = []
        det_plate = []
        poss_plate.append(gray)
        to_end = False
        max_lim = 2
        tolerance = 100

        for i in range(max_lim):
            mx_v = np.amax(poss_plate[i])
            _, gray = cv2.threshold(poss_plate[i], 0.7 * mx_v, 0.7 * mx_v, cv2.THRESH_BINARY)
            # gray = (mx_v / 255) * gray
            d00 = poss_plate[i] - gray
            det_plate.append(gray)
            if np.linalg.norm(d00) < tolerance:
                break
            # I should do some morphology related things
            d00 = cv2.erode(d00, superKernel, iterations=1)
            d01 = cv2.morphologyEx(d00, cv2.MORPH_RECT, rectKernel)
            d02 = cv2.dilate(d01, pKernel, iterations=3)
            d03 = cv2.dilate(d02, rectKernel, iterations=2)
            poss_plate.append(d03)

        for i in range(len(det_plate)):
            # loop over the contours
            _, cnts, _ = cv2.findContours(det_plate[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                # grab the bounding box associated with the contour and compute the area and
                # aspect ratio
                (x, y, w, h) = cv2.boundingRect(c)
                aspectRatio = w / float(h)
                # compute the rotated bounding box of the region
                rect = cv2.minAreaRect(c)
                box = np.int0(cv2.boxPoints(rect))
                box[0, :] += 3
                box[1, :] += 4
                box[:, 0] -= 3
                box[:, 1] -= 4
                # ensure the aspect ratio, width, and height of the bounding box fall within
                # tolerable limits, then update the list of license plate regions
                # todo: lets examin the scaled factores with some tolerance
                if (aspectRatio > 2.1 and aspectRatio < 7) and h > 10 and w > 50 and h < 125 and w < 400:
                    retRegions.append(box)
                    retCoords.append((x, y, w, h))

    counter = 0
    # for every possible plate regions
    for region in retRegions:
        pltCoord = retCoords[counter]
        counter += 1
        # plate, _, _, _ = detplt.getFilteredImageFromRegion(region)
        plate = perspective.four_point_transform(gimg, region)
        plate = imutils.resize(plate, width=400)
        # show to decide best capturing
        plt.imshow(plate)  # there is a problem for closed images
        plt.close()

print('its finished')
print(retRegions)

# for more improvement i will work on the interaction between size and threshold! in order to find most proper one!
