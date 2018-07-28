#coding:utf-8
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from pylab import*


def mat_math(intput, str):
    output = intput
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if str == "atan":
                output[i, j] = math.atan(intput[i, j])
            if str == "sqrt":
                output[i, j] = math.sqrt(intput[i, j])
    return output

class CVSegmentation():

    def __init__(self, image, IniLSF):
        # Model parameters
        self.mu = 1
        self.nu = 0.003 * 255 * 255
        self.num = 20
        self.epison = 1
        self.step = 0.1
        self.LSF = IniLSF
        #Image = cv2.imread('1.bmp',1) # read the main image
        #image = cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
        img = np.array(image.copy(), dtype=np.float64) # make numpy array

        # Initial level set function
        IniLSF = np.ones((img.shape[0], img.shape[1]),img.dtype)
        IniLSF[30:80,30:80]= -1
        IniLSF=-IniLSF
        # Draw the initial outline
        #Image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        plt.figure(1),plt.imshow(image.copy()),plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
        plt.contour(IniLSF,[0],color = 'b',linewidth=2)  #画LSF=0处的等高线
        plt.draw(),plt.show(block=False)



# CV  level set classifiaction
    def CV(self, img):

        Drc = (self.epison / math.pi) / (self.epison*self.epison+ self.LSF*self.LSF)
        Hea = 0.5*(1 + (2 / math.pi)*mat_math(self.LSF/self.epison,"atan"))
        Iy, Ix = np.gradient(self.LSF)
        s = mat_math(Ix*Ix+Iy*Iy,"sqrt")
        Nx = Ix / (s+0.000001)
        Ny = Iy / (s+0.000001)
        Mxx,Nxx =np.gradient(Nx)
        Nyy,Myy =np.gradient(Ny)
        cur = Nxx + Nyy
        Length = self.nu*Drc*cur

        Lap = cv2.Laplacian(self.LSF,-1)
        Penalty = self.mu*(Lap - cur)

        s1=Hea*img
        s2=(1-Hea)*img
        s3=1-Hea
        C1 = s1.sum()/ Hea.sum()
        C2 = s2.sum()/ s3.sum()
        CVterm = Drc*(-1 * (img - C1)*(img - C1) + 1 * (img - C2)*(img - C2))

        LSF = self.LSF + step*(Length + Penalty + CVterm)
        #plt.imshow(s, cmap ='gray'),plt.show()
        return LSF

# mu = 1
# nu = 0.003 * 255 * 255
num = 20
# epison = 1
# step = 0.1
# LSF=IniLSF
img = cv2.imread()

cvss = CVSegmentation(img, LSF)

for i in range(1,num):

    LSF = cvss.CV()
    if i % 1 == 0:
        plt.imshow(img),plt.xticks([]), plt.yticks([])
        plt.contour(LSF,[0],colors='r',linewidth=2)
        plt.draw(),plt.show(block=False),plt.pause(0.01)

