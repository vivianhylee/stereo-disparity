from PIL import Image
import math
import cv2
import numpy as np
from scipy.signal import convolve2d
from graph import build_graph, segment_graph
from smooth_filter import gaussian_grid, filter_image
from random import random


def diff_rgb(img, x1, y1, x2, y2):
    r = (img[0][x1, y1] - img[0][x2, y2]) ** 2
    g = (img[1][x1, y1] - img[1][x2, y2]) ** 2
    b = (img[2][x1, y1] - img[2][x2, y2]) ** 2
    return np.sqrt(r + g + b)


def diff_grey(img, x1, y1, x2, y2):
    v = (img[x1, y1] - img[x2, y2]) ** 2
    return np.sqrt(v)


def threshold(size, const):
    return (const / size)


def calc_energy(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    return (np.sum(np.abs(sobelx)) + np.sum(np.abs((sobely)))) / (10**8)




class Segmentation(object):
    def __init__(self, image):
        self.image = image #Image.fromarray(image)

        self.image_smooth = None
        self.graph = None
        self.forest = None

        self.height, self.width = self.image.shape[: 2]
        self.neighbor = 8  # only 4 or 8
        self.sigma = 0.8  # float(sys.argv[1])
        self.K = 3000  # float(sys.argv[3])
        self.min_size = self.height * self.width / 500  # int(sys.argv[4])

        self._preprocess()


    def filter(self, image):
        length = int(math.ceil(self.sigma * 4)) + 1
        return cv2.GaussianBlur(image, ksize=(length, length), sigmaX=self.sigma)


    def _preprocess(self):
        if len(self.image) == 3:
            r, g, b = self.image.split()

            r = self.filter(r)
            g = self.filter(g)
            b = self.filter(b)

            self.image_smooth = (r, g, b)
            self.diff = diff_rgb

        else:
            self.image_smooth = self.filter(self.image)
            self.diff = diff_grey


    def run_segmentation(self):
        self.graph = build_graph(img=self.image_smooth,
                                 width=self.width,
                                 height=self.height,
                                 diff=self.diff,
                                 neighborhood_8=self.neighbor == 8)

        self.forest = segment_graph(self.graph, self.height * self.width, self.K, self.min_size, threshold)


    def generate_image(self):
        random_color = lambda: (int(random() * 255), int(random() * 255), int(random() * 255))
        colors = [random_color() for i in xrange(self.width * self.height)]

        output = np.zeros((self.width, self.height, 3), dtype=np.float32)
        #img = Image.new('RGB', (self.width, self.height))
        #im = img.load()
        for row in xrange(self.height):
            for col in xrange(self.width):
                comp = self.forest.find(row * self.width + col)
                output[col, row] = colors[comp]

        output = np.rot90(output, k=3)
        output = np.fliplr(output)
        #return ima.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)
        return output


    def gen_comp_map(self, segmentation):
        img = np.zeros((self.height, self.width))

        comp_id = {}
        id = 1

        for r in xrange(self.height):
            for c in xrange(self.width):
                comp = tuple(segmentation[r, c])
                if comp not in comp_id:
                    comp_id[comp] = id
                    id += 1
                img[r, c] = comp_id[comp]
        return img, id - 1



if __name__ == '__main__':
    image = cv2.imread('stereo pair images/Aloe/view1.png', 0)
    sg = Segmentation(image)
    sg.run_segmentation()
    result = sg.generate_image()
    cv2.imwrite('sg_result.png', result)
