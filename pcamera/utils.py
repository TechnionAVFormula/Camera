import cv2 as cv
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
        self.image = cv.imread(self.imageFilePath)

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
        HSV_image = copy.copy(self.image)
        hsv = cv.cvtColor(HSV_image, cv.COLOR_BGR2HSV)
        # channels
        ch1, ch2, ch3 = cv.split(hsv)
        cv.imshow('original', self.image)
        # cv.imshow('HSV - h', ch1)
        cv.imshow('HSV - s', ch2)
        # cv.imshow('HSV - v', ch3)
        # print(hsv)
        hist_ch2 = cv.calcHist(ch2, [0], None, [256], [0,256])
        # hist_ch3 = cv.calcHist(ch3, [0], None, [256], [0,256])
        plt.plot(hist_ch2)
        plt.show()
        pass


class Images():
    def __init__(self):
        self.images = 0

    def loadImagesFromFolder(self, folderPath):
        pass

    def totalHistogram(self):
        pass


path = 'yolo_cones/data/Combo_img/in5_0030'
image = Image(path)
image.imageHistogram()
print('Done')