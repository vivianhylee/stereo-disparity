import cv2
import os
import numpy as np

class EvaluationDM(object):
    def __init__(self, imageGT, imageET, saveMap=False, rootdir=''):
        self.imageGT = self._preprocess(imageGT)

        self.imageET = []
        for image in imageET:
            self.imageET.append(self._preprocess(image))

        self.saveMap = saveMap
        self.rootdir = rootdir

        self.numRow, self.numCol = self.imageGT.shape
        self.numSample = len(self.imageET)
        self.report = None

    def __str__(self):
        return 'BMP: %5.2f, MAE: %5.2f' %(self.report[0, 0], self.report[1, 0])


    def _preprocess(self, image):
        if len(image.shape) == 2:
            return image.astype(np.float32)
        else:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)


    def evaluate(self):
        self.report = np.zeros((2, self.numSample))
        for i, disparity in enumerate(self.imageET):
            self.report[0, i] = self.BMP(disparity)
            self.report[1, i] = self.MAE(disparity)

            if self.saveMap:
                map = self.errorMap(disparity)
                cv2.imwrite(os.path.join(self.rootdir, 'errorMap' + str(i) + '.png'), map)


    def errorMap(self, disparity, threshold=5):
        #return np.absolute((self.imageGT - disparity))
        map = np.zeros(self.imageGT.shape, dtype=np.uint8)
        map[np.where(np.absolute(self.imageGT - disparity) > threshold)] = 255
        #map = cv2.normalize(map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return map


    def BMP(self, disparity, threshold=5):
        '''
        Bad Matched Pixels (%)
        '''
        e = len(np.where(np.absolute(self.imageGT - disparity) > threshold)[0]) / float(self.numRow * self.numCol) * 100
        return e


    def MAE(self, disparity):
        '''
        Mean Absolute Error
        '''
        e = np.mean(np.absolute((self.imageGT - disparity)))
        return e


    def MRE(self, disparity):
        return np.mean(np.absolute((self.imageGT - disparity)) / self.imageGT) #/ (self.numRow * self.numCol)


    def MSE(self, disparity):
        return np.mean((self.imageGT - disparity) ** 2) #/ (self.numRow * self.numCol)




if __name__ == '__main__':
    rootdir = 'stereo pair images/'
    filename = np.array(['GT.png', 'BM.png', 'GC.png', 'GC_fast.png', 'BM_filled.png', 'BM_filled_v1.png', 'BM_filled_v2.png', 'BM_filled_v3.png'])
    file = open(rootdir + 'performance.txt', 'w')

    for subdir, dirs, files in os.walk(rootdir):
        run, disparities = False, [None] * len(filename)

        for f in files:
            filepath = os.path.join(subdir, f)
            for i, fname in enumerate(filename):
                if filepath.endswith(fname):
                    disparities[i] = cv2.imread(filepath, 0)
                    run = True

        if run:
            # print subdir
            evaluator = EvaluationDM(imageGT=disparities[0], imageET=disparities[1: ])
            evaluator.evaluate()
            result = evaluator.report
            # print 'Bad pixel %: ', result[0]
            # print 'MAE        : ', result[1]
            # print

            file.write(subdir + "\n")
            file.write(np.array2string(filename[1: ]) + "\n")
            file.write("BMP(%)\n" + np.array2string(result[0]) + "\n")
            file.write("MAE(%)\n" + np.array2string(result[1]) + "\n")
            file.write("\n")

    print "result saved to file at: " + rootdir
    file.close()


