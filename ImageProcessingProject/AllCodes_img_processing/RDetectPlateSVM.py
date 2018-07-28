#
# Car Detection part
#

import cv2
import numpy as np
import dlib
from threading import Thread

class RDetectPlateSVM:
    def __init__(self, clssFile = "/path to car data", res = (320,240)):
        ''' class initialization
        :parameter clFile: classification file
        :parameter (,) res: desired resolution
        '''

        self.plates = []
        self.__img = None
        self.__resolution = res

        # load the detector
        # self.detector = dlib.simple_object_detector("car_front_detector.svm")
        self.detector = dlib.simple_object_detector(clssFile)

    def getPlates(self):
        return self.plates

    def detect(self, img):
        ''' detect car algorithm
        :parameter cv2.Mat img: current image
        '''
        # img = cv2.GaussianBlur(img, (9, 9), 3)
        self.__img = img
        self.plates.clear()
        x0 = img.shape[1]
        y0 = img.shape[0]
        # self.__detect()

        # self.__img = cv2.resize(img, self.__resolution)
        x1 = self.__img.shape[1]
        y1 = self.__img.shape[0]
        boxes = self.detector(cv2.cvtColor(self.__img, cv2.COLOR_BGR2RGB))

        # scaling for fitting different size of image
        for b in boxes:
            (x, y, w, h) = (b.left(), b.top(), b.right(), b.bottom())
            x2 = int(x * x0 / x1)
            y2 = int(y * y0 / y1)
            w2 = int(w * x0 / x1)
            h2 = int(h * y0 / y1)
            self.plates.append((x2, y2, w2, h2))

        return self.plates


from imutils import paths


if __name__ == '__main__':
    path = "/path to images/"

    imgs = sorted(list(paths.list_images(path)), reverse=True)

    detector = RDetectPlateSVM()


    for path in imgs:
        imgOriginal = cv2.imread(path)

        image = cv2.resize(imgOriginal, (1280,720))

        plates = detector.detect(image)

        for (x, y, w, h) in plates:
            # midX = int((x + w)/2)
            # midY = int((y + h)/2)
            cv2.rectangle(image, (x, y), (w, h), (0, 0, 255), 2)

        cv2.imshow("image", image)
        cv2.waitKey(0)
