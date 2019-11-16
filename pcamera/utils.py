import cv2
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
import numpy as np


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
        pass


class Images():
    def __init__(self):
        self.images = 0

    def loadImagesFromFolder(self, folderPath):
        pass

    def totalHistogram(self):
        pass


path = 'Cone Detection/yolo_cones/data/dataset/in5_0030'
image = Image(path)
print('Done')
