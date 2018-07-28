#
#
import sklearn as sk
import cv2
import RLpDefinitions as definitions
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
import pickle
import os
import RDetectPlates as DetPlt
import RLpPreprocess as preprocess

from sklearn.neural_network import MLPClassifier


def datacll(grayplt, plate):
    _, cnts, _ = cv2.findContours(plate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # plate.copy()
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]
    character = []
    pltchar = {}# Precomputation
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        # rect = cv2.minAreaRect(c)
        # box = np.int0(cv2.boxPoints(rect))
        #
        M = cv2.moments(c)

        cX = int(M["m10"] / M["m00"])
        #cY = int(M["m01"] / M["m00"])
        #
        #
        #
        if h > (definitions.MIN_PIXEL_HEIGHT-18) and w > definitions.MIN_PIXEL_WIDTH:
            xlb = np.array([0, x - 3]).max();
            xrb = np.array([grayplt.shape[1], x + w + 3]).min()
            ylb = np.array([0, y - 3]).max();
            yrb = np.array([grayplt.shape[0], y + h + 3]).min()
            cl_img = grayplt[ylb: yrb, xlb: xrb]
            tr_img = plate[ylb: yrb, xlb: xrb]
            # platel = perspective.four_point_transform(grayplt, box)
            #final1 = cv2.dilate(tr_img, None, iterations=1)

            plt.imshow(cl_img)
            plt.imshow(grayplt)
            timg = cv2.threshold(cl_img, 100, 255, cv2.THRESH_BINARY_INV)[1]
            plt.imshow(timg)

            cv2.drawContours(grayplt, [c], -1, (0, 255, 0), 2)
            #cv2.circle(grayplt, (cX, cY), 1, (255, 255, 255), -1)
            #plt.imshow(plate)

            #m,n = tr_img.shape

            clssname = input('inpu class name:')

            img_data = timg.reshape(-1)
            img_data[-1] = clssname

            print('this char is in the class of: ',img_data[-1])

            character.append(img_data)
            pltchar.update({clssname: cX})

    print('red plate: ', sorted(pltchar, key= pltchar.__getitem__))


    return character

# path to input images

path = '/'

train = False




if __name__ == "__main__":
    #img = cv2.imread()
    # vectorizing the above code and adding the label to the end of it

    imgs = sorted(list(paths.list_images(path)), reverse=True)

    for path in imgs:
        plt.close()
        #           print(path)
        #           imgPath = path
        #rnd = np.random.randint(0, len(imgs) - 1, 1)[0]
        #imgPath = imgs[rnd]

        imgOriginal = cv2.imread(path)
        plt.imshow(imgOriginal)
        # plt.close()
        lpDetector = DetPlt.RDetectPlates(imgOriginal)
        plates, _ = lpDetector.detectPossiblePlateRegions()
        #plates = lpDetector.detectPlates()

        for plate in plates:
            grayplate, processed_plate = preprocess.kmeans_seg(plate)
            characters = datacll(grayplate, processed_plate)

            if os.path.exists(path + '/img_data'):
                # "with" statements are very handy for opening files.
                with open(path + '/img_data', 'rb') as rfp:
                    olddata = pickle.load(rfp)
            else: # this will happen at the first time only or changing the drive
                with open(path + '/img_data', 'wb') as wfp:
                    pickle.dump(characters, wfp)


            #first_name = input("Please enter your name:")
            #score = input("Please enter your score:")

            #high_scores = first_name, score
            olddata.append(characters)
            # Now we "sync" our database
            with open(path + '/img_data', 'wb') as wfp:
                pickle.dump(olddata, wfp)

            # Re-load our database
            #with open(path + '/img_data', 'rb') as rfp:
            #    chardata = pickle.load(rfp)




if train == True:

    mlp = MLPClassifier(hidden_layer_sizes=(20, 3), max_iter=150, alpha=1e-4,
                        solver='sgd', verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init=.1)

    # training multilayer perception
    # simply feed learnset and labels
    #mlp.fit(learnset, learnlabels)
