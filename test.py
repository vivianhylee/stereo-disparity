import cv2
import time
import numpy as np
from skimage import segmentation, color, filters



#img = data.coffee()
img = cv2.imread('data_scene_flow/training/image_2/000001_10.png', 0)
t1 = time.time()
labels1 = segmentation.slic(img, compactness=0.1, n_segments=500, sigma=0.8)
out1 = color.label2rgb(labels1, img, kind='avg')

# g = graph.rag_mean_color(img, labels1, mode='similarity')
# labels2 = graph.cut_normalized(labels1, g)
# out2 = color.label2rgb(labels2, img, kind='avg')


# labels2 = segmentation.felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
# out2 = color.label2rgb(labels2, img, kind='avg')

# labels3 = segmentation.quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
# out3 = color.label2rgb(labels3, img, kind='avg')

labels4 = segmentation.watershed(filters.sobel(img), markers=250, compactness=0.001)
out4 = color.label2rgb(labels4, img, kind='avg')
print len(np.unique(out4))


out = np.vstack((img, out4))

cv2.imshow('r', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
#

# fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(6, 8))
#
# ax[0].imshow(img)
# ax[1].imshow(out1)
# ax[2].imshow(out2)

#
# for a in ax:
#     a.axis('off')
#
# plt.tight_layout()