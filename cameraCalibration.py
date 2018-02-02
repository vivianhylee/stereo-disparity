import numpy as np
import cv2
import glob



def readImage(folder, extension='jpg'):
    image_stack = []
    for fname in glob.glob(folder + '/*.' + extension):
        image = cv2.imread(fname)
        image_stack.append(image)
    return image_stack


def createKnownBoardPosition(boardSize):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    numberRow, numberCol = boardSize
    worldKnownSpaceCorner = np.zeros((numberRow * numberCol, 3), np.float32)
    worldKnownSpaceCorner[:, : 2] = np.mgrid[0: numberRow, 0: numberCol].T.reshape(-1, 2)
    return worldKnownSpaceCorner


def getChessboardCorners(image, boardSize, flags=0, show=False):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(image=grayImage, patternSize=boardSize, flags=flags)

    if found:
        cv2.cornerSubPix(image=grayImage, corners=corners, winSize=(11, 11), zeroZone=(-1, -1), criteria=criteria)

        if show:
            plotImage = np.array(image, dtype=np.uint8)
            cv2.drawChessboardCorners(plotImage, boardSize, corners, found)
            cv2.imshow('image', plotImage)
            cv2.waitKey(500)
            cv2.destroyAllWindows()

    return found, corners


def cameraCalibration(calibrationImages, boardSize, squareEdgeLength, flags=(0, 0)):
    findChessboardCornersFlags, calibrationCameraFlags = flags

    imageDimension = calibrationImages[0].shape[: -1]

    worldKnownSpaceCorner = createKnownBoardPosition(boardSize=boardSize)

    objectPoints = [] # 3d point in real world space
    imagePoints = [] # 2d points in image plane.

    for image in calibrationImages:
        found, corners = \
            getChessboardCorners(image=image,
                                 boardSize=boardSize,
                                 flags=findChessboardCornersFlags,
                                 show=False)

        # If found, add object points, image points (after refining them)
        if found:
            objectPoints.append(worldKnownSpaceCorner)
            imagePoints.append(corners)

    # Calibrate Camera
    ret, intrinsicMatrix, distortionMatrix, rotationVectors, translationVectors \
        = cv2.calibrateCamera(objectPoints=objectPoints,
                              imagePoints=imagePoints,
                              imageSize=imageDimension,
                              cameraMatrix=None,
                              distCoeffs=None,
                              flags=calibrationCameraFlags)

    return intrinsicMatrix, distortionMatrix, rotationVectors, translationVectors, objectPoints, imagePoints



def imageUndistort(image, intrinsicMatrix, distortionMatrix):
    h, w = image.shape[: 2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsicMatrix, distortionMatrix, (w, h), 1, (w, h))
    dst = cv2.undistort(image, intrinsicMatrix, distortionMatrix, None, newcameramtx)
    return dst


def computeReprojectionError(intrinsicMatrix, distortionMatrix, rotationVectors, translationVectors, objectPoints, imagePoints):
    totalError = 0
    for i in xrange(len(objectPoints)):
        imagePoints2, _ = cv2.projectPoints(objectPoints[i], rotationVectors[i], translationVectors[i], intrinsicMatrix, distortionMatrix)
        error = cv2.norm(imagePoints[i], imagePoints2, cv2.NORM_L2) / len(imagePoints2)
        totalError += error
    return totalError / len(objectPoints)


def stereoCalibration(calibrationImages, imageHeight, imageWidth, boardSize, squareEdgeLength, flags=(0, 0)):
    findChessboardCornersFlags, calibrationCameraFlags = flags

    worldKnownSpaceCorner = createKnownBoardPosition(boardSize=boardSize)

    objectPoints = [] # 3d point in real world space
    imagePoints1 = [] # 2d points in image plane.
    imagePoints2 = [] # 2d points in image plane.

    for image in calibrationImages:
        image1, image2 = image[:, : imageWidth, :], image[:, imageWidth:, :]

        found1, corners1 = \
            getChessboardCorners(image=image1,
                                 boardSize=boardSize,
                                 flags=findChessboardCornersFlags,
                                 show=False)

        found2, corners2 = \
            getChessboardCorners(image=image2,
                                 boardSize=boardSize,
                                 flags=findChessboardCornersFlags,
                                 show=False)

        # If found, add object points, image points (after refining them)
        if found1 and found2:
            objectPoints.append(worldKnownSpaceCorner)
            imagePoints1.append(corners1)
            imagePoints2.append(corners2)

    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    ret, intrinsicMatrix1, distortionMatrix1, intrinsicMatrix2, distortionMatrix2, R, T, E, F \
        = cv2.stereoCalibrate(objectPoints=objectPoints,
                              imagePoints1=imagePoints1,
                              imagePoints2=imagePoints2,
                              imageSize=(imageHeight, imageWidth),
                              flags=calibrationCameraFlags,
                              criteria=criteria)

    return intrinsicMatrix1, distortionMatrix1, intrinsicMatrix2, distortionMatrix2, R, T, E, F



def stereoRectify(imageHeight, imageWidth, intrinsicMatrix1, distortionMatrix1, intrinsicMatrix2, distortionMatrix2, R, T, flags=0):
    R1, R2, P1, P2, Q, roi1, roi2 = \
        cv2.stereoRectify(cameraMatrix1=intrinsicMatrix1,
                          cameraMatrix2=intrinsicMatrix2,
                          distCoeffs1=distortionMatrix1,
                          distCoeffs2=distortionMatrix2,
                          imageSize=(imageWidth, imageHeight),
                          R=R,
                          T=T,
                          flags=flags,
                          alpha=-1,
                          newImageSize=(0, 0))

    return R1, R2, P1, P2, Q


def computeRectifyMap(imageHeight, imageWidth, intrinsicMatrix, distortionMatrix, R, P):
    map1, map2 = cv2.initUndistortRectifyMap(intrinsicMatrix, distortionMatrix, R, P, (imageWidth, imageHeight), m1type=cv2.CV_32FC1)
    return map1, map2


def imageRemap(image, map1, map2):
    return cv2.remap(src=image, map1=map1, map2=map2, interpolation=cv2.INTER_LANCZOS4)


def imageRectification(calibrationImages, image, boardSize, squareEdgeLength, flags):
    findChessboardCornersFlags, calibrationCameraFlags, stereoRectifyFlags = flags

    imageHeight, imageWidth = calibrationImages[0].shape[0], calibrationImages[0].shape[1] / 2

    # Stereo Calibration
    intrinsicMatrix1, distortionMatrix1, intrinsicMatrix2, distortionMatrix2, R, T, E, F = \
        stereoCalibration(calibrationImages=calibrationImages,
                          imageHeight=imageHeight,
                          imageWidth=imageWidth,
                          boardSize=boardSize,
                          squareEdgeLength=squareEdgeLength,
                          flags=(findChessboardCornersFlags, calibrationCameraFlags))

    # Get rotation and projection matrix
    R1, R2, P1, P2, Q= stereoRectify(imageHeight=imageHeight,
                                     imageWidth=imageWidth,
                                     intrinsicMatrix1=intrinsicMatrix1,
                                     distortionMatrix1=distortionMatrix1,
                                     intrinsicMatrix2=intrinsicMatrix2,
                                     distortionMatrix2=distortionMatrix2,
                                     R=R,
                                     T=T,
                                     flags=stereoRectifyFlags)

    # Reproject images
    imageHeight, imageWidth = image.shape[0], image.shape[1] / 2
    image1 = image[:, : imageWidth, :]
    image2 = image[:, imageWidth: , :]

    map1x, map1y = computeRectifyMap(imageHeight=imageHeight,
                                   imageWidth=imageWidth,
                                   intrinsicMatrix=intrinsicMatrix1,
                                   distortionMatrix=distortionMatrix1,
                                   R=R1,
                                   P=P1)

    map2x, map2y = computeRectifyMap(imageHeight=imageHeight,
                                     imageWidth=imageWidth,
                                     intrinsicMatrix=intrinsicMatrix2,
                                     distortionMatrix=distortionMatrix2,
                                     R=R2,
                                     P=P2)

    imageRec1 = imageRemap(image=image1, map1=map1x, map2=map1y)
    imageRec2 = imageRemap(image=image2, map1=map2x, map2=map2y)

    return imageRec1, imageRec2



def monoCameraCalibration(imageStack, image, boardSize, squareEdgeLength, flags):
    intrinsicMatrix, distortionMatrix, rotationVectors, translationVectors, objectPoints, imagePoints \
        = cameraCalibration(calibrationImages=imageStack,
                            boardSize=boardSize,
                            squareEdgeLength=squareEdgeLength,
                            flags=flags)

    error \
        = computeReprojectionError(intrinsicMatrix=intrinsicMatrix,
                                   distortionMatrix=distortionMatrix,
                                   rotationVectors=rotationVectors,
                                   translationVectors=translationVectors,
                                   objectPoints=objectPoints,
                                   imagePoints=imagePoints)

    imageRec = imageUndistort(image, intrinsicMatrix, distortionMatrix)

    fx, fy, cx, cy = intrinsicMatrix[0, 0], intrinsicMatrix[1, 1], intrinsicMatrix[0, 2], intrinsicMatrix[1, 2]
    print intrinsicMatrix
    print
    print 'fx: ', int(fx)
    print 'fy: ', int(fy)
    print 'cx: ', int(cx)
    print 'cy: ', int(cy)
    print 'Center of image: (540, 960)'
    print 'Reprojection error: ', error

    return imageRec


def anaglyph(imageLeft, imageRight, invert=False):
    b2,g2,r1 = cv2.split(imageLeft)
    b1,g1,r2 = cv2.split(imageRight)
    dst = cv2.merge((b2, g2, r2)) if invert else cv2.merge((b1, g1, r1))
    return dst





if __name__ == '__main__':
    folder = 'calibration1121'

    squareEdgeLength = 1
    boardSize = (6, 8)

    findChessboardCornersFlags = 0
    calibrationCameraFlags = cv2.CALIB_SAME_FOCAL_LENGTH + cv2.CALIB_ZERO_TANGENT_DIST
    stereoRectifyFlags = cv2.CALIB_ZERO_DISPARITY

    # Read in images
    imageStack = readImage(folder=folder)
    targetImage = imageStack[5]

    # Mono camera calibration
    # imageRect = monoCameraCalibration(imageStack=imageStack,
    #                                   image=targetImage,
    #                                   boardSize=boardSize,
    #                                   squareEdgeLength=squareEdgeLength,
    #                                   flags=(findChessboardCornersFlags, calibrationCameraFlags))

    # Analyph
    image1, image2 = \
        imageRectification(calibrationImages=imageStack,
                           image=targetImage,
                           boardSize=boardSize,
                           squareEdgeLength=squareEdgeLength,
                           flags=(findChessboardCornersFlags, calibrationCameraFlags, stereoRectifyFlags))

    anaglyphImage = anaglyph(imageLeft=image1, imageRight=image2)


