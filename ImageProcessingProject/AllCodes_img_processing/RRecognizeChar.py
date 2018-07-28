#
# character recognizer
#

import RLpDefinitions as definitions

from blockbinarypixelsum import BlockBinaryPixelSum

cPickleForPython3 = True

try:
    cPickleForPython3 = False
    import cPickle
except:
    cPickleForPython3 = True
    import _pickle as cPickle

clsPath = definitions.clsPath

if cPickleForPython3:
    char_classifier = clsPath+"adv_char.cpickle"
    digit_classifier = clsPath+"adv_digit.cpickle"
    all_classifier = clsPath+"adv_all.cpickle"
else:
    char_classifier = clsPath+"py2_results/adv_char.cpickle"
    digit_classifier = clsPath+"py2_results/adv_digit.cpickle"
    all_classifier = clsPath+"py2_results/adv_all.cpickle"

# load the character and digit classifiers
charModel = cPickle.loads(open(char_classifier,"rb").read())
digitModel = cPickle.loads(open(digit_classifier,"rb").read())
allModel = cPickle.loads(open(all_classifier,"rb").read())

# initialize the descriptor
blockSizes = ((30, 30), (30, 60), (60, 30), (60, 60))
desc = BlockBinaryPixelSum(targetSize=(180, 90), blockSizes=blockSizes)

# this is where we apply the actual char recognition
def recognizeCharsInPlate(lp):
    strChars = ""               # this will be the return value, the chars in the lic plate
    text = ""

    listOfMatchingChars = lp.charList
    imgThresh = lp.imgFiltered

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # sort chars from left to right

    count = 0
    for count, currentChar in enumerate(listOfMatchingChars):
        # crop char out of threshold image
        imgROI = imgThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                 currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        #cv2.imshow("c "+str(count), imgROI)

        char = imgROI
        features = desc.describe(char).reshape(1, -1)

        prediction = ""
        if lp.charTypeList[count] == "number":
            prediction = digitModel.predict(features)[0]
        if lp.charTypeList[count] == "letter":
            prediction = charModel.predict(features)[0]
        if lp.charTypeList[count] == "both":
            prediction = allModel.predict(features)[0]

        # if count < 2:
        #     prediction = digitModel.predict(features)[0]
        # if count >= 2 and count < 4:
        #     prediction = charModel.predict(features)[0]
        # if count >= 4:
        #     prediction = digitModel.predict(features)[0]

        # prediction = allModel.predict(features)[0]

        # update the text of recognized characters
        text += prediction.upper()

    return text
