#
# plate localization
#

# for sunny days

#import tensorflow as tf
import cv2
import imutils
import numpy as np
from imutils import paths
#import RDetectPlates as detplt
from imutils import perspective
import matplotlib.pyplot as plt


#import RLpPreprocess as prpr

#
# trying different images by addressing different paths
#
path = '/path to image data/'
imgs = sorted(list(paths.list_images(path)), reverse=True)
rnd = np.random.randint(0, len(imgs)-1, 1)[0]
#rnd = 39
# testing the detector:
det_1 = 'fh'
det_2 = 'cmp'


# convexhull extractıon: mergıng closed contours
# the measure for evaluatıng closedness of the contours>

def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 25:
                return True
            elif i==row1-1 and j==row2-1:
                return False

# convexhull drawıng system
def convexhull(contours):
    LENGTH = len(contours)
    status = np.zeros((LENGTH,1))

    for i,cnt1 in enumerate(contours):
        x = i
        if i != LENGTH-1:
            for j,cnt2 in enumerate(contours[i+1:]):
                x = x+1
                dist = find_if_close(cnt1,cnt2)
                if dist == True:
                    val = min(status[i],status[x])
                    status[x] = status[i] = val
                else:
                    if status[x]==status[i]:
                        status[x] = i+1

    unified = []
    maximum = int(status.max())+1
    for i in range(maximum):
        pos = np.where(status==i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            unified.append(hull)

    return unified



run = True
# the goal of this part is to find plate!!
while(run):
    run = False
    # for first path
    #imgOrg = cv2.imread(path)   #imgOrg = np.asarray(img)

    # for second path
    imgOrg = cv2.imread(imgs[rnd]) # imgs[rnd]
    #prpr.preprocess(img)
    #s_x, s_y, ch = img.shape
    #intface = paths.list_images(path) # list()
    #imgOrg = sorted(intface, reverse=True)

    plt.imshow(imgOrg)
    plt.close()
    try:
        gimg = cv2.cvtColor(imgOrg, cv2.COLOR_BGR2GRAY)
    except:
        print('there is an error in making img gray')
    #plt.imshow(gimg)
    #plt.close()
#
# working on Fatihs
# this part should be checked once again
#

    detector = 'fh'
    if detector == det_1:

        retRegions = []  # this will be the return value
        retCoords = []  # this will be the return value
        # # defining kernels
        #
        # Vertical Kernels
        vertKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
        pKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        #
        # Horizontal Kernels
        bKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        b2Kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
        smallKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 3))
        HKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 4))  # the rectangle kernel
        superKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))  # 27,3 check 29 also
        #
        # poss_plate = []
        # det_plate = []
        # bigpics = []
        # # poss_plate.append(gray)
        # to_end = False
        # max_lim = 10
        # tolerance = 100
        #
        # # for i in range(max_lim):
        # # convert the image to grayscale, and apply the blackhat operation
        # img_gray = cv2.cvtColor(self.imgOriginal, cv2.COLOR_BGR2GRAY)
        # # step one
        #
        # gradX = np.absolute(cv2.Sobel(img_gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1))
        # (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        # gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
        #
        # # I used one name for readability and memory usage
        # # step two
        # gray = cv2.medianBlur(cv2.blur(cv2.GaussianBlur(gradX, (15, 15), 10), (15, 15)), 15)
        # gray = cv2.erode(gray, superKernel, iterations=1)
        # gray = cv2.erode(gray, rectKernel, iterations=3)
        # gray = cv2.dilate(gray, rectKernel, iterations=2)
        # gray = cv2.GaussianBlur(gray, (5, 5), 10)
        # mx_v = np.amax(gray)
        # _, gray = cv2.threshold(gray, 0.3 * mx_v, mx_v, cv2.THRESH_BINARY)
        # gray = cv2.dilate(gray, smallKernel, iterations=10)
        #
        # _, cnts, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #
        # #
        # for cnt in cnts:
        #     #
        #     RecPnt = np.int0(cv2.boxPoints(cv2.minAreaRect(cnt)))
        #     bigpics.append(RecPnt)
            #

        bigpics = []  # this will be the return value
        #retCoords = []  # this will be the return value

        # initialize the rectangular and square kernels to be applied to the image,
        # then initialize the list of license plate regions
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))

        # convert the image to grayscale, and apply the blackhat operation
        #img_gray = cv2.cvtColor(self.imgOriginal, cv2.COLOR_BGR2GRAY)

        blackhat = cv2.morphologyEx(gimg, cv2.MORPH_BLACKHAT, rectKernel)

        # find regions in the image that are light
        light = cv2.morphologyEx(gimg, cv2.MORPH_CLOSE, rectKernel)
        light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY)[1]


        # compute the Scharr gradient representation of the blackhat image and scale the
        # resulting image into the range [0, 255]
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

        # blur the gradient representation, apply a closing operation, and threshold the
        # image using Otsu's method
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        gradX = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # perform a series of erosions and dilations on the image
        gradX = cv2.erode(gradX, squareKernel, iterations=2)
        gradX = cv2.dilate(gradX, squareKernel, iterations=3)


        # take the bitwise 'and' between the 'light' regions of the image, then perform
        # another series of erosions and dilations
        thresh = cv2.bitwise_and(gradX, gradX, mask=light)
        thresh = cv2.erode(thresh, squareKernel, iterations=2)
        thresh = cv2.dilate(thresh, squareKernel, iterations=2)


        # find contours in the thresholded image
        _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
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
            if (aspectRatio > 2 and aspectRatio < 8) and h > 10 and w > 50 and h < 125 and w < 400:
                # if h > 10 and w > 50 and h < 250 and w < 750:
                bigpics.append(box)
                #retCoords.append((x, y, w, h))

        for bigpic in bigpics:
            plt.close()
            recheck = True
            gray = perspective.four_point_transform(gimg, bigpic)
            #    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #         #gray = rotateImage(gray, cnt)
            #         #gray = imutils.resize(gray, width=400)
            plt.imshow(gray)
            plt.close()

            # Left filter:
            # noise_removal = cv2.bilateralFilter(gray, 9, 75, 75)
            # equal_histogram = cv2.equalizeHist(noise_removal)
            # morph_image = cv2.morphologyEx(equal_histogram, cv2.MORPH_OPEN, rectKernel, iterations=5)
            gradX = np.absolute(cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1))
            gradY = np.absolute(cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1))
            #gradX = np.absolute(cv2.Sobel(gradX, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1))
            (minVal, maxVal) = (np.min(gradX), np.max(gradX))
            gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
            gray_proc = cv2.medianBlur(cv2.blur(cv2.GaussianBlur(gradX, (9, 9), 9), (9, 9)), 9)

            plt.imshow(gray_proc)
            plt.close()
            # cv2.imshow('gray proc', gray_proc)
            # cv2.waitKey(30)

            # left filters
            left_gray = cv2.erode(gray_proc, rectKernel, iterations=1)
            left_gray = cv2.morphologyEx(left_gray, cv2.MORPH_RECT, smallKernel)
            left_gray = cv2.morphologyEx(left_gray, cv2.MORPH_RECT, rectKernel)
            left_gray = cv2.dilate(left_gray, vertKernel, iterations=1)
            left_gray = cv2.dilate(left_gray, rectKernel, iterations=3)
            _, left_gray = cv2.threshold(left_gray, 0, 255, cv2.THRESH_OTSU)
            #
            plt.imshow(left_gray)
            plt.close()
            # # cv2.imshow('left gray', left_gray)
            # # cv2.waitKey(30)
            # # middle filter
            #
            mid_gray = cv2.dilate(gray_proc, HKernel, iterations=3)
            mid_gray = cv2.dilate(mid_gray, rectKernel, iterations=1)
            _, mid_gray = cv2.threshold(mid_gray, 0, 255, cv2.THRESH_OTSU)
            #
            plt.imshow(mid_gray)
            plt.close()
            # # cv2.imshow('mid gray', mid_gray)
            # # cv2.waitKey(30)
            # left_gray = cv2.morphologyEx(gray_proc, cv2.MORPH_RECT, rectKernel)
            # left_gray = cv2.morphologyEx(gray_proc, cv2.MORPH_RECT, rectKernel)
            # left_gray = cv2.dilate(left_gray, pKernel, iterations=2)
            # left_gray = cv2.dilate(left_gray, vertKernel, iterations=1)
            # left_gray = cv2.dilate(left_gray, bKernel, iterations=1)
            #
            # mx_v = np.amax(left_gray)
            # _, left_gray = cv2.threshold(left_gray, 0.35 * mx_v, mx_v, cv2.THRESH_BINARY)
            #  right side removing:
            # # right fılter
            right_gray = cv2.dilate(gray_proc, HKernel, iterations=8)
            _, right_gray = cv2.threshold(right_gray, 0, 255, cv2.THRESH_OTSU)
            right_gray = cv2.morphologyEx(right_gray, cv2.MORPH_RECT, rectKernel)
            # right_gray = cv2.morphologyEx(right_gray, cv2.MORPH_RECT, rectKernel)
            #
            plt.imshow(right_gray)
            plt.close()
            # # cv2.imshow('right gray', right_gray)
            #
            # # right_gray = cv2.morphologyEx(gray_proc,  cv2.MORPH_RECT, superKernel)
            # # right_gray = cv2.morphologyEx(right_gray, cv2.MORPH_RECT, superKernel)
            # # right_gray = cv2.morphologyEx(right_gray, cv2.MORPH_RECT, smallKernel)
            # # right_gray = cv2.dilate(right_gray, HKernel, iterations=8)
            # # #right_gray = cv2.dilate(right_gray, smallKernel, iterations=2)
            # # mx_v = np.amax(right_gray)
            # # _, right_gray = cv2.threshold(right_gray, 0.35 * mx_v, mx_v, cv2.THRESH_BINARY)
            #
            # # summing up masks:
            finalg_l = cv2.bitwise_and(gray, gray, mask=left_gray)
            # finalg_r = cv2.bitwise_and(gray, gray, mask=right_gray)
            # finalg_m = cv2.bitwise_and(gray, gray, mask=mid_gray)
            finalg = cv2.bitwise_and(finalg_l, finalg_l, mask=mid_gray)
            finalg = cv2.bitwise_and(finalg, finalg, mask=right_gray)
            #
            plt.imshow(finalg)
            plt.close()
            # do not use this
            # finalg = cv2.bilateralFilter(finalg, 9, 75, 75)
            # finalg = cv2.equalizeHist(noise_removal)
            # gradX = np.absolute(cv2.Sobel(finalg, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1))

            mx_v = np.amax(finalg)
            _, finalg = cv2.threshold(finalg, 0.72 * mx_v, mx_v, cv2.THRESH_BINARY)
            finalg = cv2.Canny(finalg, 0.90 * mx_v, mx_v)

            # todo: should decide about this parameter
            finalg = cv2.dilate(finalg, bKernel, iterations=2)

            plt.imshow(finalg)
            plt.close()

            _, cnts, _ = cv2.findContours(finalg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



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
                        x += bigpic[0, 0]
                        y += bigpic[1, 1]
                        box[:, 0] += bigpic[0, 0]
                        box[:, 1] += bigpic[1, 1]
                        retRegions.append(box)
                        retCoords.append((x, y, w, h))
                        recheck = False
                        plate = perspective.four_point_transform(imgOrg, box)
                        plt.imshow(plate)
                        plt.close()

            # if len(cnts)>1 and recheck:
            elif len(cnts) > 1 and len(cnts) < 10:
                cvxhs = convexhull(cnts)
                for cvxh in cvxhs:
                    (x, y, w, h) = cv2.boundingRect(cvxh)
                    aspectRatio = w / float(h)
                    # compute the rotated bounding box of the region
                    rect = cv2.minAreaRect(cvxh)
                    box = np.int0(cv2.boxPoints(rect))

                    if (aspectRatio > 2.5 and aspectRatio < 7) and h > 10 and w > 50 and h < 125 and w < 400:
                        x += bigpic[0, 0]
                        y += bigpic[1, 1]
                        box[:, 0] += bigpic[0, 0]
                        box[:, 1] += bigpic[1, 1]
                        retRegions.append(box)
                        retCoords.append((x, y, w, h))
                        plate = perspective.four_point_transform(imgOrg, box)
                        plt.imshow(plate)
                        plt.close()

