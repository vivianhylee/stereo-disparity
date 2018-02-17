import numpy as np
import cv2
from matplotlib import pyplot as plt

def drawMatches(img1, kp1, img2, kp2, matches):
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    # Create the output image
    # The rows of the output are the largest between the two images
    # and the columns are simply the sum of the two together
    # The intent is to make this a colour image, so make this 3 channels
    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

    # Place the first image to the left
    out[:rows1, :cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2, cols1:] = np.dstack([img2, img2, img2])
    disparity = 0
    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        if int(y1) != int(y2):
            continue
        print abs(x1 - x2)

        disparity = max((abs(x1 - x2)), disparity)

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2) + cols1, int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0), 1)
    print 'a= ', disparity
    return out




img1 = cv2.imread('view1.png')
img2 = cv2.imread('view5.png')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.uint8)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.uint8)

img1 = cv2.normalize(img1 * 1.1, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
img2 = cv2.normalize(img2 * 1.1, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


# Initiate SIFT detector
# orb = cv2.ORB()

# # find the keypoints and descriptors with SIFT
# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)
#
# # create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
# # Match descriptors.
# matches = bf.match(des1,des2)

sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

 # create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)#[: len(matches) / 2]

img3 = drawMatches(img1,kp1,img2,kp2,matches)

plt.imshow(img3, 'gray'),plt.show()

