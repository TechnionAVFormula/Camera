import cv2
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
import numpy as np
import colorsys
import copy


class Image():
    def __init__(self, filePath):
        self.imageFilePath = filePath + '.png'
        self.boxFilePath = filePath + '.txt'
        self.image = 0
        self.boxPoints = 0
        self.loadImage(False)
        self.loadBoxPoints()
        self.plotImageWithBox()

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

    def imageHistogram(self):
        cv2.imshow("Original image before HSV", self.image)
        RGB_image = copy.copy(self.image)

        hsv = cv2.cvtColor(RGB_image, cv2.COLOR_BGR2HSV)
        blur_hsv = cv2.GaussianBlur(hsv, (5,5), 0)
        # channels
        ch1, ch2, ch3 = cv2.split(blur_hsv)
        # range blue color
        blue_color_l = (0,0,0)
        blue_color_d = (240,270,100)
        
        mask = cv2.inRange(blur_hsv, blue_color_l, blue_color_d)
        new_S = cv2.bitwise_and(blur_hsv, blur_hsv, mask=mask)
        cv2.imshow('blue', new_S)
        hist_ch2 = cv2.calcHist(ch2, [0], None, [256], [0,256])
        plt.plot(hist_ch2)
        plt.show()
        pass

    def imageShowRectRGB(self):
        imgB = copy.copy(self.image)
        imgY = copy.copy(self.image)

        blueLow = np.array([60, 0, 0])
        blueHigh = np.array([230, 90, 50])
        yellowLow = np.array([0, 80, 100])
        yellowHigh  = np.array([50, 220, 240])

        maskBlue = cv2.inRange(imgB, blueLow, blueHigh)
        maskYellow = cv2.inRange(imgY, yellowLow, yellowHigh)
        outputBlue = cv2.bitwise_and(imgB, imgB, mask = maskBlue)
        outputYellow = cv2.bitwise_and(imgY, imgY, mask = maskYellow)
        
        # show the images
        cv2.imshow("Blue", outputBlue)
        cv2.imshow("Yellow", outputYellow)
        
        tempBlue = cv2.cvtColor(outputBlue, cv2.COLOR_BGR2GRAY) 
        tempYellow = cv2.cvtColor(outputYellow, cv2.COLOR_BGR2GRAY)
        edgedBlue = cv2.Canny(tempBlue, 30, 200)
        edgedYellow = cv2.Canny(tempYellow, 30, 200) 

        contoursB, hierarchyB = cv2.findContours(edgedBlue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        contours_polyB = [None]*len(contoursB)
        boundRectB = [None]*len(contoursB)

        contoursY, hierarchyY = cv2.findContours(edgedYellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        contours_polyY = [None]*len(contoursY)
        boundRectY = [None]*len(contoursY)

        for i, c in enumerate(contoursB):
            contours_polyB[i] = cv2.approxPolyDP(c, 3, True)
            boundRectB[i] = cv2.boundingRect(contours_polyB[i])

        for i, c in enumerate(contoursY):
            contours_polyY[i] = cv2.approxPolyDP(c, 3, True)
            boundRectY[i] = cv2.boundingRect(contours_polyY[i])
               
        # drawingBlue = np.zeros((edgedBlue.shape[0], edgedBlue.shape[1], 3), dtype=np.uint8)
        # drawingYellow = np.zeros((edgedYellow.shape[0], edgedYellow.shape[1], 3), dtype=np.uint8)
    
        for i in range(len(contoursB)):
            color = (0, 0, 255)
            cv2.rectangle(self.image, (int(boundRectB[i][0]), int(boundRectB[i][1])), \
            (int(boundRectB[i][0]+boundRectB[i][2]), int(boundRectB[i][1]+boundRectB[i][3])), color, 2)

        for i in range(len(contoursY)):
            color = (0, 0, 255)
            cv2.rectangle(self.image, (int(boundRectY[i][0]), int(boundRectY[i][1])), \
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


path = 'yolo_cones/data/Combo_img/in5_0030'
image = Image(path)
# image.imageHistogram()
image.imageShowRectRGB()
print('Done')