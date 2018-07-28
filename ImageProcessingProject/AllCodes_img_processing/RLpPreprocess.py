import cv2
import numpy as np
import RLpDefinitions as definitions
import RDetectPlates


from RPossibleChar import RPossibleChar as PossibleChar

import imutils
from imutils import perspective
import matplotlib.pyplot as plt

GAUSSIAN_SMOOTH_FILTER_SIZE = (7, 7)
ADAPTIVE_THRESH_BLOCK_SIZE = 39
ADAPTIVE_THRESH_WEIGHT = 3

# ADAPTIVE_THRESH_BLOCK_SIZE = 31
# ADAPTIVE_THRESH_WEIGHT = 3

from skimage.filters import threshold_local
import imutils

def preprocess(imgOriginal):
    imgOriginal2 = imgOriginal.copy()
    #imgOriginal = cv2.bilateralFilter(imgOriginal, 9, 120, 120)
    # i chnged
    fig, ax = plt.subplots(9, 2, figsize=(10, 10), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
    ax[0, 0].imshow(imgOriginal2) #     fig, ax = plt.subplots(9, 2, figsize=(10, 10), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
    #ax[0, 0].imshow(mark_boundaries(imgOriginal2, imgOriginal2))
    ax[0, 0].set_title("original img")
    #ax[0, 0].set_title("original img")

    count = 1
    cv2.imwrite('plate' + count,  imgOriginal2 )

    #plt.subplot(911)
    #plt.imshow(imgOriginal2)

    imgGrayscale = cv2.bilateralFilter(imgOriginal, 9, 120, 120) # not good, to be removed
    # #imgGrayscale = extractValue(imgOriginal)
    # plt.subplot(912)
    # plt.imshow(imgGrayscale)
    # fig, ax = plt.subplots(9, 2, figsize=(10, 10), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
    # ax[0, 0].imshow( imgOriginal2)
    # ax[0, 0].set_title("Felzenszwalbs's method")


    #segments_fz = felzenszwalb(imgGrayscale, scale=10, sigma=0.1, min_size=2)
    # plt.subplot(211)
    # plt.imshow(segments_fz)
    # plt.subplot(212)
    # plt.imshow(plate)
    # fig, ax = plt.subplots(9, 2, figsize=(10, 10), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
    # ax[0, 0].imshow(mark_boundaries(imgOriginal2, imgOriginal2))
    # ax[0, 0].set_title("Felzenszwalbs's method")
    #
    # plt.subplot(913)
    # plt.imshow(segments_fz)




    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)

    plt.subplot(914)
    plt.imshow(imgMaxContrastGrayscale)

    imgBlurred = cv2.GaussianBlur(imgGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    imgThresh = cv2.dilate(imgThresh, rectKernel, iterations=1)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgThresh = cv2.erode(imgThresh, rectKernel, iterations=2)
    # cv2.imshow("imgThresh", imgThresh)
    plt.subplot(915)
    plt.imshow(imgThresh)

    #img = cv2.cvtColor(imgOriginal2,cv2.COLOR_BGR2GRAY)
    # i chnged
    img = imgOriginal2
    img = cv2.GaussianBlur(img, (7,7), 0)
    img = cv2.equalizeHist(img)

    plt.subplot(916)
    plt.imshow(img)     # this part works better

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,5))
    # img = cv2.morphologyEx(img,cv2.MORPH_DILATE,kernel,iterations=3)
    thr = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)[1]
    thr = cv2.bitwise_not(thr)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,3))
    img = cv2.morphologyEx(img,cv2.MORPH_ERODE,kernel,iterations=1)
    # cv2.imshow("equalizeHist thr", thr)


    plt.subplot(917)
    plt.imshow(img)

    #V = cv2.split(cv2.cvtColor(imgOriginal2, cv2.COLOR_BGR2HSV))[2]
    V = imgOriginal2
    T = threshold_local(V, 25, offset=19, method="gaussian")
    thresh = (V > T).astype("uint8") * 255
    thresh = cv2.bitwise_not(thresh)
    # resize the license plate region to a canonical size
    thresh = imutils.resize(thresh, width=400)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=3)
    # cv2.imshow("bitwise_not Thresh", thresh)


    plt.subplot(918)
    plt.imshow(thresh) # this part is not good at all !!


    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
    #gray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

    blackhat = cv2.morphologyEx(imgOriginal2, cv2.MORPH_BLACKHAT, rectKernel) # gray replaced by imageoriginal
    blackhat = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)[1]
    # rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    # blackhat = cv2.erode(blackhat, rectKernel, iterations=1)
    # cv2.imshow("blackhat",blackhat)

    plt.subplot(919)
    plt.imshow(blackhat)  # this part is not works well

    #
    adthr = cv2.GaussianBlur(imgOriginal2, (7,7), 0) # gray replaced by imageoriginal
    adthr = cv2.equalizeHist(adthr)
    adthr = cv2.adaptiveThreshold(adthr, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 15)
    adthr = cv2.bitwise_not(adthr)
    # gray = cv2.erode(gray, rectKernel, iterations=1)
    # cv2.imshow("adaptiveThreshold gray",adthr)
    plt.subplot(921)
    plt.imshow(adthr)

    # cv2.waitKey(0)

    return imgOriginal, imgGrayscale, imgThresh, adthr#imgThresh#blackhat#cv2.bitwise_or(thr, gray)

def preprocess2(imgOriginal):

    ########
    imgOriginalFiltered = cv2.bilateralFilter(imgOriginal.copy(), 9, 120, 120)
    imgMaxContrastGrayscale = extractValue(imgOriginalFiltered)

    # imgMaxContrastGrayscale = cv2.cvtColor(imgOriginalFiltered, cv2.COLOR_BGR2GRAY)

    thr = cv2.threshold(imgMaxContrastGrayscale, 180, 255, cv2.THRESH_BINARY)[1]
    thr = cv2.bitwise_not(thr)
    lap = cv2.Laplacian(thr, cv2.CV_8UC1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
    thr = cv2.erode(thr, kernel, iterations=1)
    cv2.imshow("thrrr",thr)
    im2, cnts, hierarchy = cv2.findContours(thr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    if len(cnts)>0:
        contour = cnts[0]
        for cnt in cnts:
            if cv2.contourArea(cnt) > cv2.contourArea(contour):
                contour = cnt

        rect = cv2.minAreaRect(contour)
        box = np.int0(cv2.boxPoints(rect))

        plate = perspective.four_point_transform(imgOriginal, box)
        plate = imutils.resize(plate, width=400)

        cv2.imshow("rplt", plate)
    else:
        plate = imgOriginal
    plate = imgOriginal
    #######

    imgOriginalFiltered = cv2.bilateralFilter(plate.copy(), 9, 120, 120)
    imgGrayscale = extractValue(imgOriginalFiltered)

    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)

    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    imgThresh = cv2.dilate(imgThresh, rectKernel, iterations=1)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgThresh = cv2.erode(imgThresh, rectKernel, iterations=1)

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    blackhat = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)[1]
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    blackhat = cv2.erode(blackhat, rectKernel, iterations=1)
    # imgThresh = cv2.adaptiveThreshold(blackhat, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    return plate, imgGrayscale, imgThresh, imgThresh

def extractValue(imgOriginal):

    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgValue
#
def maximizeContrast(imgGrayscale):

     height, width = imgGrayscale.shape
     structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

     imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
     imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)
     imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
     imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

     return imgGrayscalePlusTopHatMinusBlackHat


def kmeans_seg(image):
    imgOriginal = image.copy()
    #imgOriginal = cv2.bilateralFilter(imgOriginal, 9, 120, 120)
    # i chnged
    xxlKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    xlKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    xsKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    ylKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    ysKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    #
    fig, ax = plt.subplots(5, 2, figsize=(10, 10), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
    ax[0, 0].imshow(imgOriginal); ax[0, 0].set_title("original img")#     fig, ax = plt.subplots(9, 2, figsize=(10, 10), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})

    # X - direction
    gX = np.absolute(cv2.Sobel(imgOriginal, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1))
    gX = (255 * ((gX - gX.min()) / (gX.max() - gX.min()))).astype("uint8")
    t0 = cv2.GaussianBlur(gX, (5, 5), 0)
    erdx = cv2.dilate(cv2.erode(t0, xsKernel, iterations=10), xlKernel, iterations=12)  # carefully setting parameters
    #
    tr_x = cv2.threshold(erdx, .65 * erdx.max(), erdx.max(), cv2.THRESH_TRUNC)[1]
    #
    erdx = cv2.threshold(tr_x, .7 * tr_x.max(), tr_x.max(), cv2.THRESH_BINARY)[1]
    ax[1, 0].imshow(erdx); ax[1, 0].set_title("erdx gbX")

    # # Y - direction
    gY = np.absolute(cv2.Sobel(imgOriginal, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1))
    gY = (255 * ((gY - gY.min()) / (gY.max() - gY.min()))).astype("uint8")
    #t1 = cv2.GaussianBlur(gY, (5, 5), 0)
    #erdy = cv2.dilate(cv2.erode(t1, xlKernel, iterations=3), ylKernel, iterations=5)
    #erdyd = cv2.dilate(erdy, ylKernel, iterations=3)
    #tr_y = cv2.threshold(erdyd, .65 * erdyd.max(), erdyd.max(), cv2.THRESH_TRUNC)[1]
    #erdy = cv2.threshold(tr_y, .65 * tr_y.max(), tr_y.max(), cv2.THRESH_BINARY)[1]
    #erdy = cv2.dilate(erdy, ylKernel, iterations=10)
    #ax[2, 0].imshow(erdy);     ax[2, 0].set_title("erdy ")

    #if np.linalg.norm(gY) < 10000 or np.linalg.norm(gX) < 10000:
    print('most probably this is not plate: norm y{}, norm x {}'.format(np.linalg.norm(gY), np.linalg.norm(gX)))

    # a filter for borders
    #mx_v = np.amax(erdy);
    #_, t4 = cv2.threshold(erdy, .4 * mx_v, mx_v, cv2.THRESH_BINARY_INV + cv2.THRESH_TRUNC)
    #mx_v = np.amax(t4);
    #_, filtery = cv2.threshold(t4, .4 * mx_v, mx_v, cv2.THRESH_BINARY_INV)
    #filtery = cv2.dilate(filtery, xlKernel, iterations=3)
    #ax[2, 1].imshow(filtery); ax[2, 1].set_title("filtery 5-10 ")


    # using the filter to remove borders
    #s_marker = cv2.bitwise_and(erdx.astype(np.int32), erdx.astype(np.int32), mask=erdyd)
    # erdxy1 = cv2.bitwise_and(erdx, erdx, mask=erdy)
    # erdxye = cv2.erode(erdxy1, None, iterations=2)
    # # we reched to the ımportant fılter
    # erdxy = cv2.dilate(erdxye, None, iterations=5)
    # ax[2, 0].imshow(erdxy); ax[2, 0].set_title("gradxy 5-10 gbX")

    # lets see the propertıes of the contours:
    #cnts = cv2.findContours(erdxy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

    # compute the convexhull
    #cxcnt = convexhull(cnts)
    #for cvxh in cxcnt:
    #    (x, y, w, h) = cv2.boundingRect(cvxh)
    #    rect = cv2.minAreaRect(cvxh)
    #    box = np.int0(cv2.boxPoints(rect))
    #    pplate = perspective.four_point_transform(imgOriginal, box)
    #    plt.subplot()
    #    plt.imshow(pplate)


    # new segmentation

    #img_float = np.float32(imgOriginal)  # Convert image from unsigned 8 bit to 32 bit float
    #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
    # Defining the criteria ( type, max_iter, epsilon )
    # cv2.TERM_CRITERIA_EPS - stop the algorithm iteration if specified accuracy, epsilon, is reached.
    # cv2.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations, max_iter.
    # cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER - stop the iteration when any of the above condition is met.
    #max_iter = 100
    # cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER - stop the iteration when any of the above condition is met.
    # max_iter - An integer specifying maximum number of iterations.In this case it is 10
    # epsilon - Required accuracy.In this case it is 1
    #k = 8  # Number of clusters
    #ret, label, centers = cv2.kmeans(img_float, k, None, criteria, 50, cv2.KMEANS_RANDOM_CENTERS)
    # apply kmeans algorithm with random centers approach
    #center = np.uint8(centers)
    # Convert the image from float to unsigned integer
    #res = center[label.flatten()]
    # This will flatten the label
    #res2 = res.reshape(image.shape)
    # Reshape the image
    # making image colored --
    backtorgb = cv2.cvtColor(imgOriginal, cv2.COLOR_GRAY2RGB)
    meanshift = cv2.pyrMeanShiftFiltering(backtorgb, sp=8, sr=16, maxLevel=1, \
                                          termcrit=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1))
    # Apply meanshift algorithm on to image
    #ax[3, 0].imshow(meanshift); ax[3, 0].set_title("meanshift result img")

    imgOriginal = cv2.cvtColor(meanshift, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(imgOriginal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # apply thresholding to convert the image to binary
    #ax[4, 0].imshow(thresh); ax[4, 0].set_title("bin otsu threshold")
    fg = cv2.erode(thresh, None, iterations=1)
    #ax[0, 1].imshow(fg); ax[0, 1].set_title("erode fg")
    # erode the image
    bgt = cv2.dilate(thresh, None, iterations=1)
    #ax[1, 1].imshow(bgt); ax[1, 1].set_title("dilate bgt")
    # Dilate the image
    bg = cv2.threshold(bgt, 1, 100, 1)[1]
    bg = cv2.erode(bg, None, iterations=1)
    bg[1:8, :] = 0
    bg[-7:, :] = 0
    bg[:, 1:8] = 0
    bg[:, -7:] = 0

    ax[2, 1].imshow(bg); ax[2, 1].set_title("threshold erod of bgt")
    # Apply thresholding

    return image, bg


def extra():
    # these parameters should be set
    final1 = cv2.bitwise_and(bg, bg, mask=erdx)
    final1 = cv2.erode(final1, ysKernel, iterations=2)
    final1 = cv2.dilate(final1, ylKernel, iterations=7)
    final1 = cv2.erode(final1, xsKernel, iterations=2)
    final1 = cv2.dilate(final1, xlKernel, iterations=2)

    ax[3, 1].imshow(final1); ax[3, 1].set_title("final 1 result ")

    marker = cv2.add(fg, bg)
    #Add foreground and background
    canny = cv2.Canny(marker, 110, 150)
    ax[6, 0].imshow(canny); ax[6, 0].set_title("canny of marker")
    #Apply canny edge detector
    new, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #Finding the contors in the image using chain approximation
    marker32 = np.int32(marker)
    #converting the marker to float 32 bit
    cv2.watershed(backtorgb, marker32.astype(np.int32))
    ax[7, 0].imshow(backtorgb); ax[7, 0].set_title("watershed result")
    #Apply watershed algorithm
    m = cv2.convertScaleAbs(marker32)
    thresh = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    ax[3, 1].imshow(thresh); ax[3, 1].set_title("bin otsu threshold of m")

    #Apply thresholding on the image to convert to binary image
    thresh_inv = cv2.bitwise_not(thresh)
    ax[9, 0].imshow(thresh_inv); ax[9, 0].set_title("thresh_inv")
    #Invert the thresh
    res = cv2.bitwise_and(imgOriginal, imgOriginal, mask=thresh)
    ax[5, 1].imshow(res); ax[5, 1].set_title("bitwise and result mask thresh")
    #Bitwise and with the image mask thresh
    res3 = cv2.bitwise_and(imgOriginal, imgOriginal, mask=thresh_inv)
    ax[6, 1].imshow(res3); ax[6, 1].set_title("bitwise and result mask thresh_inv")
    #Bitwise and the image with mask as threshold invert
    res4 = cv2.addWeighted(res, 1, res3, 1, 0)
    ax[7, 1].imshow(res4); ax[7, 1].set_title("add weighted result img")
    #Take the weighted average
    final = cv2.drawContours(res4, contours, -1, (0, 255, 0), 1)
    #Draw the contours on the image with green color and pixel width is 1
    ax[8, 1].imshow(final); ax[8, 1].set_title("final result img")

    #Watershed segmentation with distance metric algorithm: -----------------------
    #using the preprocessed image or doing some preprocessing here
    #morph, erosion, dilation, etc

    thresh2 = cv2.threshold(imgOriginal, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    #ax[0, 1].imshow(thresh2); ax[0, 1].set_title("thresh2 dis-w")

    final2 = cv2.bitwise_and(thresh2, thresh2, mask=erdx)
    final2 = cv2.erode(final2, None, iterations=2)
    final2 = cv2.dilate(final2, None, iterations=2)

    ax[4, 1].imshow(final2); ax[4, 1].set_title("final 2 result ")

    #noise removal
    kernel = np.ones((2, 2), np.uint8)
    # opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    ax[1, 1].imshow(closing); ax[1, 1].set_title("morhp- closing")
    # sure background area
    sure_bg = cv2.dilate(closing, kernel, iterations=3)

    ax[2, 1].imshow(sure_bg); ax[2, 1].set_title("sure bg")

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)

    ax[3, 1].imshow(closing); ax[3, 1].set_title("dist_transform")

    # Threshold
    sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)[1]
    ax[4, 1].imshow(sure_fg);  ax[4, 1].set_title("sure fg")


    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ax[5, 1].imshow(unknown);  ax[5, 1].set_title("unknown ")

    # Marker labelling
    markers = cv2.connectedComponents(sure_fg)[1]

    ax[6, 1].imshow(markers); ax[6, 1].set_title("connected marker of fg")

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(backtorgb, markers)
    backtorgb[markers == -1] = [255, 0, 0]

    ax[7, 1].imshow(markers); ax[7, 1].set_title("watershed res img")
    plt.close()
    # I can design norm based filter here !!! norm of grad x and y which works well
    plt.imshow(bg)
