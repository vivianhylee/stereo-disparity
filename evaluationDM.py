import cv2
import numpy as np

class EvaluationDM(object):
    def __init__(self, imageGT, imageET):
        self.imageGT = self._preprocess(imageGT)
        self.imageET = self._preprocess(imageET)

        self.numRow, self.numCol = self.imageGT.shape


    def _preprocess(self, image):
        if len(image.shape) == 2:
            return image.astype(np.float32)
        else:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)


    def BMP(self, threshold=10):
        '''
        Bad Matched Pixels (%)
        '''
        numRow, numCol = self.imageGT.shape
        return len(np.where(np.absolute(self.imageGT - self.imageET) > threshold)[0]) / float(numRow * numCol) * 100


    def MAE(self):
        '''
        Mean Absolute Error
        '''
        return np.mean(np.absolute((self.imageGT - self.imageET))) #/ (self.numRow * self.numCol)


    def MRE(self):
        return np.mean(np.absolute((self.imageGT - self.imageET)) / self.imageGT) #/ (self.numRow * self.numCol)


    def MSE(self):
        return np.mean((self.imageGT - self.imageET) ** 2) #/ (self.numRow * self.numCol)

