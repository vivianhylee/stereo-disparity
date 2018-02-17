import cv2
import numpy as np
from disparityMap import DisparityMap
from evaluationDM import EvaluationDM
from segmentation import run_segmentation
from skimage import segmentation, filters
from main import fill_occlusion
import time
from main import scaleDown

cap = cv2.VideoCapture('videoplayback.mp4')
# cap1 = cv2.VideoCapture('BoxGroundTruth/Short_L.avi')
# cap2 = cv2.VideoCapture('BoxGroundTruth/Short_R.avi')
# cap3 = cv2.VideoCapture('BoxGroundTruth/Short_GroundTruth.avi')
#
i = 0
while cap.isOpened(): #(cap1.isOpened() and cap2.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # ret2, frame2 = cap2.read()
    # ret3, frame3 = cap3.read()
    # print i
    i += 1
    if ret == True:# and ret2 == True:
        # imageL, imageR = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # imageGT = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

        height, width = frame.shape[: 2]
        imageL, imageR = frame[:, : width / 2, :], frame[:, width / 2: , :]
        imageL = cv2.cvtColor(imageL, cv2.COLOR_BGR2GRAY)[50: -50]
        imageR = cv2.cvtColor(imageR, cv2.COLOR_BGR2GRAY)[50: -50]
        # imageL = scaleDown(imageL)
        # imageR = scaleDown(imageR)

        if i >= 1200 and i <= 1900:
            # cv2.imwrite('image1_' + str(i) + '.png', imageL)
            # cv2.imwrite('image2_' + str(i) + '.png', imageR)

            comp_map = segmentation.watershed(filters.sobel(imageL), markers=100, compactness=0.00001)

            dm = DisparityMap(imageLeft=imageL, imageRight=imageR, ndisparities=48)
            disparity = dm.semiGlobalBlockMatch()

            disparity_filled = fill_occlusion(disparity=disparity, segmentation=comp_map)
            disparity_filled = cv2.normalize(disparity_filled, alpha=0, beta=255, norm_type=cv2.cv.CV_MINMAX, dtype=cv2.cv.CV_8U)
            disparity_filled = cv2.medianBlur(disparity_filled, 3)

            # Display the resulting frame
            cv2.imshow('Frame', np.hstack((imageL, disparity_filled)))

            # Press Q on keyboard to  exit
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    # Break the loop
    else:
        break
        # cap1 = cv2.VideoCapture('BoxGroundTruth/Short_L.avi')
        # cap2 = cv2.VideoCapture('BoxGroundTruth/Short_R.avi')
        # cap3 = cv2.VideoCapture('BoxGroundTruth/Short_GroundTruth.avi')

cap.release()

# cap2.release()
# cap3.release()

cv2.destroyAllWindows()
