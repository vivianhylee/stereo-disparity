import cv2
import cv2.cv as cv
import numpy as np


class DisparityMap(object):
    def __init__(self, imageLeft, imageRight, ndisparities=0):
        self.imageL = imageLeft
        self.imageR = imageRight

        self.grayL = self._preprocess(self.imageL)
        self.grayR = self._preprocess(self.imageR)

        self.ndisparities = self._computeNumberOfDisparities() #self._getNumberOfDisparities(ndisparities)
        print 'Image info: ', self.imageL.shape, self.ndisparities

    def _preprocess(self, image):
        if len(image.shape) == 2:
            return image.astype(np.uint8)
        else:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)


    def _getNumberOfDisparities(self, ndisparities):
        d = min(ndisparities, self.grayL.shape[1] / 4) if ndisparities != 0 else ndisparities
        return (d / 16) * 16


    def _computeNumberOfDisparities(self):
        #orb = cv2.ORB()
        orb = cv2.SIFT()

        kp1, des1 = orb.detectAndCompute(self.grayL, None)
        kp2, des2 = orb.detectAndCompute(self.grayR, None)

        #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[: len(matches) / 2]

        disparity = 0
        for match in matches:
            img1_idx = match.queryIdx
            img2_idx = match.trainIdx

            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt

            if int(y1) - int(y2) != 0:
                continue

            disparity = max(disparity, abs(x1 - x2))

        disparity = min(int(disparity * 2), self.grayL.shape[1] / 4)
        return (disparity / 16) * 16


    def pureBlockMatch(self, SADWindowSize=5):
        stereo = cv2.StereoBM(preset=cv2.cv.CV_STEREO_BM_BASIC, ndisparities=self.ndisparities, SADWindowSize=SADWindowSize)

        """
        presetspecifies the whole set of algorithm parameters, one of:

        BASIC_PRESET - parameters suitable for general cameras
        FISH_EYE_PRESET - parameters suitable for wide-angle cameras
        NARROW_PRESET - parameters suitable for narrow-angle cameras
        """

        disparity = stereo.compute(self.grayL, self.grayR)
        #disparity_visual = cv2.normalize(disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return disparity


    def semiGlobalBlockMatch(self):
        """
        A typical correspondence algorithm consists of following three stages:

        Pre-filtering to normalize the image brightness and enhance texture
        Matching points along horizontal lines in local windows
        Post-filtering to eliminate bad matches
        """
        SADWindowSize = 3

        stereo = cv2.StereoSGBM()

        stereo.minDisparity = 0
        stereo.numberOfDisparities = self.ndisparities  # 96
        stereo.SADWindowSize = SADWindowSize  # 13
        stereo.disp12MaxDiff = 1  # -1
        stereo.speckleWindowSize = 100  # 0  # 350
        stereo.speckleRange = 32  # 2
        stereo.P1 = SADWindowSize * SADWindowSize * 8   # 4 * numChannel * SADWindowSize * SADWindowSize
        stereo.P2 = SADWindowSize * SADWindowSize * 32  # 32 * numChannel * SADWindowSize * SADWindowSize
        stereo.uniquenessRatio = 10
        stereo.preFilterCap = 63
        stereo.fullDP = True

        disparity = stereo.compute(self.grayL, self.grayR) / 16.
        disparity = disparity - np.amin(disparity)

        return disparity


    def graphCut(self, maxIters=4):
        numRow, numCol = self.grayL.shape

        disparityLeft  = cv.CreateMat(numRow, numCol, cv.CV_16S)
        disparityRight = cv.CreateMat(numRow, numCol, cv.CV_16S)

        stereo = cv.CreateStereoGCState(self.ndisparities, maxIters)
        cv.FindStereoCorrespondenceGC(left=cv.fromarray(self.grayL),
                                      right=cv.fromarray(self.grayR),
                                      dispLeft=disparityLeft,
                                      dispRight=disparityRight,
                                      state=stereo,
                                      useDisparityGuess=0)

        disparity_visual = -1 * np.array(disparityLeft)
        #disparity_visual = cv2.normalize(disparity_visual, alpha=0, beta=255, norm_type=cv2.cv.CV_MINMAX, dtype=cv2.cv.CV_8U)

        return disparity_visual










if __name__ == '__main__':

    pass