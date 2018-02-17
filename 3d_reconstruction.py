import cv2
import numpy as np
from point_cloud import PointCloud
from disparityMap import DisparityMap
from main import scaleDown, gen_disparity



f = 7315.238
cx = 997.555
cy = 980.754
doffs = 809.195
Tx = 380.135 / 1000.

image1 = cv2.imread('stereo pair images2/Jadeplant-perfect/im0.png')
image2 = cv2.imread('stereo pair images2/Jadeplant-perfect/im1.png')
image1_scale = scaleDown(image1, 4)
image2_scale = scaleDown(image2, 4)
image1_gray = cv2.cvtColor(image1_scale, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2_scale, cv2.COLOR_BGR2GRAY)

disparity = gen_disparity(imageL=image1_gray, imageR=image2_gray, ndisparities=280)
disparity = np.array(disparity, dtype=np.uint8)
cv2.imwrite('disparity.png', disparity)

Q = np.float32([[1., 0., 0., -cx], [0., 1., 0., -cy], [0., 0., 0., f], [0., 0., 1. / Tx, doffs / Tx]])
points = cv2.reprojectImageTo3D(disparity=disparity, Q=Q)
colors = cv2.cvtColor(image1_scale, cv2.COLOR_BGR2RGB)

points = PointCloud(coordinates=points, colors=colors)
points = points.filter_infinity()
points.write_ply('result.ply')