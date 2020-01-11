import cv2
import matplotlib.pyplot as plt
from PIL import Image as pilimage
import sys, os
import pandas as pd
import matplotlib.patches as patches
import matplotlib.image as mpimg
import numpy as np
import colorsys
import copy
import os
print(cv2.__version__)

class Image():
    def __init__(self, filePath):
        self.imageFilePath = filePath + '.png'
        self.boxFilePath = filePath + '.txt'
        self.image = 0
        self.idan_helper = []
        self.boxPoints = 0
        self.loadImage(False)
        self.loadBoxPoints()
        self.plotImageWithBox()
        self.classifiedImage = 0

    def loadImage(self, showFlag):
        self.image = cv2.imread(self.imageFilePath)

    def loadBoxPoints(self):
        """loadBoxPoints loads the boxes cooridantes into a struct
        """
        self.boxPoints = pd.read_csv(self.boxFilePath, delimiter=' ').values
        pass

    def plotImageWithBox(self):
        """plotImageWithBox plots the image with the annoted boxes from files.

        """
        fig, ax = plt.subplots(1)
        ax.imshow(self.image)
        _width = self.image.shape[0]
        _length = self.image.shape[1]
        count = 0
        for i in self.boxPoints:
            count += 1
            _boxpoints = i[1:5]
            _class = i[0]
            if _class == 0:
                color = 'blue'
            else:
                color = 'orange'
            x = _boxpoints[0]*_length - _boxpoints[2]*_length/2
            y = _boxpoints[1]*_width - _boxpoints[3]*_width/2
            rect = patches.Rectangle(
                (x, y), _boxpoints[2]*_length, _boxpoints[3]*_width, linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        plt.show()
        print(count)
        pass

    def imageMask(self):
        for i in range(550,self.image.shape[0]):
            for j in range(200,1120):
                self.image[i][j] = (0,0,0)
            pass
        pass

        for i in range(400,550):
            for j in range(450,900):
                self.image[i][j] = (0,0,0)
            pass
        pass

        for i in range(0,200):
            for j in range(self.image.shape[1]):
                self.image[i][j] = (0,0,0)
            pass
        pass

        for i in range(self.image.shape[0]):
            for j in range(0,200):
                self.image[i][j] = (0,0,0)
            pass
        pass

        for i in range(self.image.shape[0]):
            for j in range(1100,1280):
                self.image[i][j] = (0,0,0)
            pass
        pass

        #cv2.imshow("masked image", self.image)
        #cv2.waitKey(0)

    def imageHistogram(self):
        cv2.imshow("Original image before HSV", self.image)
        rgbImage = copy.copy(self.image)

        hsv = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2HSV)
        blur_hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        # channels
        ch1, ch2, ch3 = cv2.split(blur_hsv)
        # range blue color
        blue_color_l = (0, 0, 0)
        blue_color_d = (240, 270, 100)

        mask = cv2.inRange(blur_hsv, blue_color_l, blue_color_d)
        new_S = cv2.bitwise_and(blur_hsv, blur_hsv, mask=mask)
        cv2.imshow('blue', new_S)
        hist_ch2 = cv2.calcHist(ch2, [0], None, [256], [0, 256])
        plt.plot(hist_ch2)
        plt.show()
        pass


    def imageHSV_ratio(self, showColorPlot=False):
        imgB = self.image
        imgY = self.image

        cv2.split(imgB)
        tempR = imgB[:,:,0]
        tempG = imgB[:,:,1]
        tempB = imgB[:,:,2]

        for i in range(220, 450):
            for j in range(320, 1020):
                ratioR = tempR[i][j]
                ratioG = tempG[i][j]
                ratioB = tempB[i][j]
                try:
                    ratioT_B = float(ratioB) / float(ratioR + ratioB + ratioG)
                    ratioT_G = float(ratioG) / float(ratioR + ratioB + ratioG)
                    ratioT_R = float(ratioR) / float(ratioR + ratioB + ratioG)
                    if (ratioT_B < 0.7 and ratioT_G < 0.85 and ratioT_R <= 1):
                        self.image[i][j] = (0,0,0)
                except ZeroDivisionError:
                    ratioT = 0
            pass
        pass

        selfGrey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        contoursB, hierarchyB = cv2.findContours(selfGrey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_polyB = []
        boundRectB = []
        areasB = np.array([cv2.contourArea(countour) for countour in contoursB])
        for i, c in enumerate(contoursB):
            contours_polyB.append(cv2.approxPolyDP(c, 3, True))
            if areasB[i] > 30 and areasB[i] < 140:
               boundRectB.append(cv2.boundingRect(contours_polyB[i]))
               self.idan_helper.append(cv2.boundingRect(contours_polyB[i]))
        for i in range(len(boundRectB)):
            color = (0, 0, 255)
            cv2.rectangle(self.image, (int(boundRectB[i][0]), int(boundRectB[i][1])),
              (int(boundRectB[i][0]+boundRectB[i][2]), int(boundRectB[i][1]+boundRectB[i][3])), color, 2)

        cv2.imshow('image',self.image)
        
        cv2.waitKey(0)

        print(contours_polyB[5])

        pass


    def imageShowRectHSV(self, minimumRectangleArea=5, showColorPlot=False):
        imgB = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        imgY = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        blueLow = np.array([90, 50, 50])
        blueHigh = np.array([150, 255, 255])
        yellowLow = np.array([20, 100, 100])
        yellowHigh = np.array([40, 255, 255])

        maskBlue = cv2.inRange(imgB, blueLow, blueHigh)
        maskBlue2 = cv2.dilate(maskBlue, np.ones((5, 5), np.uint8))
        maskBlue3 = cv2.erode(maskBlue2, np.ones((5, 5), np.uint8))
        maskYellow = cv2.inRange(imgY, yellowLow, yellowHigh)
        maskYellow2 = cv2.dilate(maskYellow, np.ones((5, 5), np.uint8))
        maskYellow3 = cv2.erode(maskYellow2, np.ones((5, 5), np.uint8))
        outputBlue = cv2.bitwise_and(imgB, imgB, mask=maskBlue3) ## TODO change to conv function
        outputYellow = cv2.bitwise_and(imgY, imgY, mask=maskYellow3) ## TODO change to conv function

        tempBlue = cv2.cvtColor(outputBlue, cv2.COLOR_BGR2GRAY)
        tempYellow = cv2.cvtColor(outputYellow, cv2.COLOR_BGR2GRAY)
        edgedBlue = cv2.Canny(tempBlue, 30, 300)
        edgedYellow = cv2.Canny(tempYellow, 30, 300)

        contoursB, hierarchyB = cv2.findContours(
            edgedBlue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_polyB = []
        boundRectB = []
        areasB = np.array([cv2.contourArea(countour) for countour in contoursB])

        contoursY, hierarchyY = cv2.findContours(
            edgedYellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        areasY = np.array([cv2.contourArea(countour) for countour in contoursY])
        contours_polyY = []
        boundRectY = []

        for i, c in enumerate(contoursB):
            contours_polyB.append(cv2.approxPolyDP(c, 3, True))
            if areasB[i] >minimumRectangleArea:
                boundRectB.append(cv2.boundingRect(contours_polyB[i]))

        for i, c in enumerate(contoursY):
            contours_polyY.append(cv2.approxPolyDP(c, 3, True))
            if areasY[i] > minimumRectangleArea:
                boundRectY.append(cv2.boundingRect(contours_polyY[i]))

        for i in range(len(boundRectB)):
            color = (0, 0, 255)
            cv2.rectangle(self.image, (int(boundRectB[i][0]), int(boundRectB[i][1])),
                          (int(boundRectB[i][0]+boundRectB[i][2]), int(boundRectB[i][1]+boundRectB[i][3])), color, 2)

        for i in range(len(boundRectY)):
            color = (0, 0, 255)
            cv2.rectangle(self.image, (int(boundRectY[i][0]), int(boundRectY[i][1])),
                          (int(boundRectY[i][0]+boundRectY[i][2]), int(boundRectY[i][1]+boundRectY[i][3])), color, 2)

        cv2.imshow('image',self.image)
        self.classifiedImage
        print(len(boundRectB))
        print(len(boundRectY))
        cv2.waitKey(0)

    def imageShowRectRGB(self):
        imgB = copy.copy(self.image)
        imgY = copy.copy(self.image)

        blueLow = np.array([60, 0, 0])
        blueHigh = np.array([230, 90, 50])
        yellowLow = np.array([0, 80, 100])
        yellowHigh = np.array([50, 220, 240])

        maskBlue = cv2.inRange(imgB, blueLow, blueHigh)
        maskYellow = cv2.inRange(imgY, yellowLow, yellowHigh)
        outputBlue = cv2.bitwise_and(imgB, imgB, mask=maskBlue)
        outputYellow = cv2.bitwise_and(imgY, imgY, mask=maskYellow)

        # show the images
        cv2.imshow("Blue", outputBlue)
        cv2.imshow("Yellow", outputYellow)

        tempBlue = cv2.cvtColor(outputBlue, cv2.COLOR_BGR2GRAY)
        tempYellow = cv2.cvtColor(outputYellow, cv2.COLOR_BGR2GRAY)
        edgedBlue = cv2.Canny(tempBlue, 30, 200)
        edgedYellow = cv2.Canny(tempYellow, 30, 200)

        contoursB, hierarchyB = cv2.findContours(
            edgedBlue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_polyB = [None]*len(contoursB)
        boundRectB = [None]*len(contoursB)

        contoursY, hierarchyY = cv2.findContours(
            edgedYellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_polyY = [None]*len(contoursY)
        boundRectY = [None]*len(contoursY)

        for i, c in enumerate(contoursB):
            contours_polyB[i] = cv2.approxPolyDP(c, 3, True)
            boundRectB[i] = cv2.boundingRect(contours_polyB[i])

        for i, c in enumerate(contoursY):
            contours_polyY[i] = cv2.approxPolyDP(c, 3, True)
            boundRectY[i] = cv2.boundingRect(contours_polyY[i])

        for i in range(len(contoursB)):
            color = (0, 0, 255)
            cv2.rectangle(self.image, (int(boundRectB[i][0]), int(boundRectB[i][1])),
                          (int(boundRectB[i][0]+boundRectB[i][2]), int(boundRectB[i][1]+boundRectB[i][3])), color, 2)

        for i in range(len(contoursY)):
            color = (0, 0, 255)
            cv2.rectangle(self.image, (int(boundRectY[i][0]), int(boundRectY[i][1])),
                          (int(boundRectY[i][0]+boundRectY[i][2]), int(boundRectY[i][1]+boundRectY[i][3])), color, 2)

        cv2.imshow('drawing', self.image)

        cv2.waitKey(0)


class Images():
    def __init__(self):
        self.images = 0

    def loadImagesFromFolder(self, folderPath):
        pass

    def totalHistogram(self):
        pass

path = "yolo_cones\data\Combo_img\in5_0001"
image = Image(path)
image.imageMask()
# image.imageShowRectHSV(minimumRectangleArea=15)
image.imageHSV_ratio()
print("check struct  ")
print(image.idan_helper)

images = []
image2 = pilimage.open('yolo_cones\data\Combo_img\in5_0001.png')
images.append(image2)
image2 = pilimage.open('yolo_cones\data\Combo_img\in5_0002.png')
images.append(image2)
image2 = pilimage.open('yolo_cones\data\Combo_img\in5_0003.png')
images.append(image2)

# images[0].save('anicircle.gif', save_all=True, append_images=images[1:], duration=100, loop=10)
# im = pilimage.open('anicircle.gif')

# image = Image('anicircle')
# image.imageShowRectRGB()
print('Done')
