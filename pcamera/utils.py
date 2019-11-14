import cv2
import matplotlib.pyplot as plt

class Image():
    def __init__(self, imageFilePath, boxFilePath):
        self.imageFilePath = imageFilePath
        self.boxFilePath = boxFilePath
        self.image = 0
    
    def loadImage(self, showFlag):
        self.image = cv2.imread(self.imageFilePath)
        if showFlag:
            plt.imshow(image)
            plt.show()

    def loadBoxPoints(self):
        """loadBoxPoints loads the boxes cooridantes into a struct
        """        
        pass

    def plotImageWithBox(self):
        """plotImageWithBox plots the image with the annoted boxes from files.
        
        """        
        pass

    def imageHistogram(self):
        pass


class Images():
    def __init__(self):
        self.images = 0

    def loadImagesFromFolder(self,folderPath):
        pass
    
    def totalHistogram(self):
        pass
