from PIL import Image
import cv2
import numpy as np
from graph import build_graph, segment_graph
from smooth_filter import gaussian_grid, filter_image
from random import random
from numpy import sqrt



def diff_rgb(img, x1, y1, x2, y2):
    r = (img[0][x1, y1] - img[0][x2, y2]) ** 2
    g = (img[1][x1, y1] - img[1][x2, y2]) ** 2
    b = (img[2][x1, y1] - img[2][x2, y2]) ** 2
    return sqrt(r + g + b)

def diff_grey(img, x1, y1, x2, y2):
    v = (img[x1, y1] - img[x2, y2]) ** 2
    return sqrt(v)


def threshold(size, const):
    return (const / size)


def build_image(forest, width, height):
    random_color = lambda: (int(random()*255), int(random()*255), int(random()*255))
    colors = [random_color() for i in xrange(width*height)]

    img = Image.new('RGB', (width, height))
    im = img.load()
    for y in xrange(height):
        for x in xrange(width):
            comp = forest.find(y * width + x)
            im[x, y] = colors[comp]

    output = img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)
    return np.array(output)


def build_map(forest, width, height):
    id = 1
    map = {}

    img = Image.new('I', (width, height))
    im = img.load()
    for y in xrange(height):
        for x in xrange(width):
            comp = forest.find(y * width + x)
            if comp not in map:
                map[comp] = id
                id += 1
            im[x, y] = map[comp]

    output = img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)
    return np.array(output)


def run_segmentation(image, print_info=False):
    image_file = Image.fromarray(image) #Image.open(filename).convert('L')
    height, width = image_file.size

    neighbor = 8  # only 4 or 8
    sigma = 0.8  # float(sys.argv[1])
    K = 3000  # float(sys.argv[3])
    min_size = height * width / 500  # int(sys.argv[4])

    grid = gaussian_grid(sigma)

    if image_file.mode == 'RGB':
        image_file.load()
        r, g, b = image_file.split()

        r = filter_image(r, grid)
        g = filter_image(g, grid)
        b = filter_image(b, grid)

        smooth = (r, g, b)
        diff = diff_rgb
    else:
        smooth = filter_image(image_file, grid)
        diff = diff_grey

    graph = build_graph(img=smooth, width=width, height=height, diff=diff, neighborhood_8=neighbor == 8)
    forest = segment_graph(graph, height * width, K, min_size, threshold)

    output = build_map(forest, width, height)
    #output = build_image(forest, width, height)

    if print_info:
        print 'Image info: ', image_file.format, height, width, image_file.mode
        print 'Number of components: %d' % forest.num_sets
    return output



if __name__ == '__main__':
    pass



    # if False:
    #     pass
    # # if len(sys.argv) != 7:
    # #     print 'Invalid number of arguments passed.'
    # #     print 'Correct usage: python main.py sigma neighborhood K min_comp_size input_file output_file'
    # else:
        # neighbor = 8 #int(sys.argv[2])
        # if neighbor != 4 and neighbor!= 8:
        #     print 'Invalid neighborhood choosed. The acceptable values are 4 or 8.'
        #     print 'Segmenting with 4-neighborhood...'
        #
        # filename = 'stereo pair images/Plastic/view1.png'
        # #391058812.0
        # #185642164.0
        # #108057354.0
        # #436118750.0
        # #126383838.0
        # #202351986.0
        # #83343882.0
        # #135749634.0
        #
        # image = cv2.imread(filename, 0)
        # print calc_energy(image)
        #
        # image_file = Image.open(filename).convert('L') #sys.argv[5])
        # size = image_file.size
        # print 'Image info: ', image_file.format, size, image_file.mode
        #
        #
        # #image_file = Image.open('imageL.png') #sys.argv[5])
        # sigma = 0.8 #float(sys.argv[1])
        # K = 3000 #float(sys.argv[3])
        # min_size = size[0] * size[1] / 500 #int(sys.argv[4])
        #
        # grid = gaussian_grid(sigma)
        #
        # if image_file.mode == 'RGB':
        #     image_file.load()
        #     r, g, b = image_file.split()
        #
        #     r = filter_image(r, grid)
        #     g = filter_image(g, grid)
        #     b = filter_image(b, grid)
        #
        #     smooth = (r, g, b)
        #     diff = diff_rgb
        # else:
        #     smooth = filter_image(image_file, grid)
        #     diff = diff_grey
        #
        # graph = build_graph(img=smooth, width=size[1], height=size[0], diff=diff, neighborhood_8=neighbor==8)
        # forest = segment_graph(graph, size[0]*size[1], K, min_size, threshold)
        #
        # image = generate_image(forest, size[1], size[0])
        # image.save('segmentImage.png')#sys.argv[6])
        #
        # print 'Number of components: %d' % forest.num_sets
        #
        # #
        # # image = cv2.imread('segmentImage.png')
        # # disparity = cv2.imread('scene_dmSGBM.png', 0)
        # #
        # # result = np.zeros(disparity.shape)
        #
        # # height, width = image.shape[: 2]
        # # comp_map, num_id = generate_comp_map(image, height, width)
        # # print num_id
        # # for id in range(1, num_id + 1):
        # #     coords = np.where(comp_map == id)
        # #     vals = disparity[coords]
        # #     #v = np.median(vals[np.where(vals > 5)])
        # #     (values, counts) = np.unique(vals[np.where(vals > 5)], return_counts=True)
        # #     if len(counts) == 0:
        # #         v = 0
        # #     else:
        # #         ind = np.argmax(counts)
        # #         v = values[ind]
        # #     result[coords] = v
        # #
        # # #result = cv2.normalize(result, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # # cv2.imwrite('result.png', result)
        #
        #
        # image = cv2.imread('segmentImage.png')
        # disparity = cv2.imread('scene_dmSGBM.png', 0)
        #
        # result = np.copy(disparity)
        #
        # height, width = image.shape[: 2]
        # comp_map, num_id = generate_comp_map(image, height, width)
        # print num_id
        # rows, cols, = np.where(disparity < 5)
        # for r, c in zip(rows, cols):
        #     id = comp_map[r, c]
        #     coords = np.where(comp_map == id)
        #     vals = disparity[coords]
        #
        #     # v = np.median(vals[np.where(vals > 5)])
        #     (values, counts) = np.unique(vals[np.where(vals > 5)], return_counts=True)
        #     if len(counts) == 0:
        #         v = 0
        #     else:
        #         ind = np.argmax(counts)
        #         v = values[ind]
        #     result[r, c] = v
        #
        # # result = cv2.normalize(result, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # cv2.imwrite('result2.png', result)
        #
        #
