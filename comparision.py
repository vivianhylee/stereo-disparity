import cv2
import os
import time
import re
import numpy as np
from multiprocessing.pool import ThreadPool
from disparityMap import DisparityMap
from evaluationDM import EvaluationDM
from segmentation import run_segmentation
from skimage import segmentation, filters, transform



def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data).astype(np.uint8)
    return data#, scale



def scaleDown(image, ratio=2):
    output = np.copy(image)
    for _ in range(ratio / 2):
        height, width = output.shape[: 2]
        output = cv2.pyrDown(output, (height / 2, width / 2))
    return output


def scaleUp(image, ratio=2):
    output = np.copy(image)
    for _ in range(ratio / 2):
        height, width = output.shape[: 2]
        output = cv2.pyrUp(output, (height * 2, width * 2))
    return output


def matchSize(image1, image2):
    h1, w1 = image1.shape[: 2]
    output = np.copy(image2)

    if h1 > output.shape[0]:
        output = np.vstack((output, np.zeros((1, output.shape[1]), dtype=np.uint8)))

    elif h1 < output.shape[0]:
        output = output[: -1, :]

    if w1 > output.shape[1]:
        output = np.hstack((output, np.zeros((output.shape[1], 1), dtype=np.uint8)))

    elif w1 < output.shape[1]:
        output = output[:, : -1]

    return output


def build_disparity(imageL, imageR, ndisparities):
    dm = DisparityMap(imageLeft=imageL, imageRight=imageR, ndisparities=ndisparities)
    disparity = dm.semiGlobalBlockMatch()
    return disparity


def build_segmentation(imageL):
    comp =segmentation.watershed(filters.sobel(imageL), markers=100, compactness=0.001)
    return comp


def fill_occlusion(disparity, segmentation):
    output = np.copy(disparity)
    index = {}
    rows, cols, = np.where(disparity == 0)
    for r, c in zip(rows, cols):
        id = segmentation[r, c]

        if id not in index:
            coords = np.where(segmentation== id)
            vals = disparity[coords]
            (values, counts) = np.unique(vals[np.where(vals > 0)], return_counts=True)

            if len(counts) == 0:
                index[id] = 0
            else:
                index[id] = values[np.argmax(counts)]

        output[r, c] = index[id]

    return output



def run():

    # rootdir = 'stereo pair images/'

    # for subdir, dirs, files in os.walk(rootdir):
    #
    #     run, imageGT, imageL, imageR, imageL_c = False, None, None, None, None
    #     ratio = 1
    #     for f in files:
    #         filepath = os.path.join(subdir, f)
    #
    #         if filepath.endswith('disp0.pfm'):
    #             imageGT = readPFM(filepath)
    #             ratio = max(imageGT.shape) / 600
    #             imageGT = scaleDown(imageGT, ratio=ratio) * (1.0/ratio)
    #             cv2.imwrite(filepath[: -3] + 'png', imageGT)
    #             run = True
    #
    #         elif filepath.endswith('im0.png'):
    #             imageL = cv2.imread(filepath, 0)
    #             imageL = scaleDown(imageL, ratio=ratio)
    #
    #         elif filepath.endswith('im1.png'):
    #             imageR = cv2.imread(filepath, 0)
    #             imageR = scaleDown(imageR, ratio=ratio)


    # for subdir, dirs, files in os.walk(rootdir):
    #
    #     run, imageGT, imageL, imageR, imageL_c = False, None, None, None, None
    #
    #     for f in files:
    #         filepath = os.path.join(subdir, f)
    #
    #         if filepath.endswith('disp1.png'):
    #             imageGT = cv2.imread(filepath, 0) * 0.5
    #             # cv2.imwrite(os.path.join(subdir, 'disparity_GT.png'), imageGT)
    #             run = True
    #
    #         elif filepath.endswith('view1.png'):
    #             imageL = cv2.imread(filepath, 0)
    #
    #         elif filepath.endswith('view5.png'):
    #             imageR = cv2.imread(filepath, 0)


    rootdir = 'data_scene_flow_12/training/'
    rootdirGT    = rootdir + 'disp_occ'
    rootdirLeft  = rootdir + 'image_0'
    rootdirRight = rootdir + 'image_1'

    for file in os.listdir(rootdirLeft):
        run, subdir, imageGT, imageLeft, imageRight = False, None, None, None, None
        if file.endswith('_10.png'):
            imageGT   = cv2.imread(os.path.join(rootdirGT,    file), 0)
            imageL    = cv2.imread(os.path.join(rootdirLeft,  file), 0)
            imageR    = cv2.imread(os.path.join(rootdirRight, file), 0)
            run = True
            subdir = rootdir + 'output/'

        if run:
            print 'Building disparity map for ' + subdir + '...\n'

            # # Semi-global block matching
            # t1 = time.time()
            # dm = DisparityMap(imageLeft=imageL, imageRight=imageR)
            # dmBM = dm.semiGlobalBlockMatch()
            #
            # print 'Semi-global BM saved, using time: ', time.time() - t1
            # # filename = file[: -4] + '_BM.png'
            # filename = 'disparity_BM.png'
            # cv2.imwrite(os.path.join(subdir, filename), dmBM)
            #
            # error = EvaluationDM(imageGT=imageGT, imageET=[dmBM])
            # error.evaluate()
            # print error.report
            # print


            # # Graph cut
            # t1 = time.time()
            # dm = DisparityMap(imageLeft=imageL, imageRight=imageR)
            # dmGC = dm.graphCut()

            # print 'GC saved, using time: ', time.time() - t1
            # filename = file[: -4] + '_GC.png'
            # # filename = 'disparity_GC.png'
            # cv2.imwrite(os.path.join(subdir, filename), dmGC)
            #
            # error = EvaluationDM(imageGT=imageGT, imageET=[dmGC])
            # error.evaluate()
            # print error.report
            # print

            # # Graph cut fast by downsize image
            # ratio = 2
            # imageL_scale = scaleDown(image=imageL, ratio=ratio)
            # imageR_scale = scaleDown(image=imageR, ratio=ratio)
            #
            # t1 = time.time()
            # dm = DisparityMap(imageLeft=imageL_scale, imageRight=imageR_scale)
            # dmGC_fast = dm.graphCut() * ratio
            # dmGC_fast = scaleUp(dmGC_fast)
            # dmGC_fast = matchSize(image1=imageGT, image2=dmGC_fast)

            # print 'GC fast saved, using time: ', time.time() - t1
            # filename = file[: -4] + '_GC_fast.png'
            # # filename = 'disparity_GC_fast.png'
            # cv2.imwrite(os.path.join(subdir, filename), dmGC_fast)
            #
            # error = EvaluationDM(imageGT=imageGT, imageET=[dmGC_fast])
            # error.evaluate()
            # print error.report
            # print
            #
            #
            # # Improve Semi-global block matching using segmentation
            # t1 = time.time()
            # comp_map = run_segmentation(image=imageL)
            # comp_map = np.asarray(comp_map)
            #
            # dm = DisparityMap(imageLeft=imageL, imageRight=imageR)
            # disparity = dm.semiGlobalBlockMatch()
            #
            # disparity_filled = fill_occlusion(disparity=disparity, segmentation=comp_map)
            #
            # print 'Semi-global BM filled saved, using time: ', time.time() - t1
            # filename = file[: -4] + '_BM_filled.png'
            # # filename = 'disparity_BM_filled.png'
            # cv2.imwrite(os.path.join(subdir, filename), disparity_filled)
            #
            # error = EvaluationDM(imageGT=imageGT, imageET=[disparity_filled])
            # error.evaluate()
            # print error.report
            # print
            #
            #
            # # Improve Semi-global block matching using segmentation coded by skimage
            # t1 = time.time()
            # # comp_map = segmentation.slic(imageL_c, compactness=0.5, n_segments=500)
            # comp_map = segmentation.slic(imageL, compactness=0.1, n_segments=500, sigma=0.8)
            #
            # dm = DisparityMap(imageLeft=imageL, imageRight=imageR)
            # disparity = dm.semiGlobalBlockMatch()
            #
            # disparity_filled_v1 = fill_occlusion(disparity=disparity, segmentation=comp_map)
            #
            # print 'Semi-global BM filled v1 saved, using time: ', time.time() - t1
            # filename = file[: -4] + '_BM_filled_v1.png'
            # # filename = 'disparity_BM_filled_v1.png'
            # cv2.imwrite(os.path.join(subdir, filename), disparity_filled_v1)
            #
            # error = EvaluationDM(imageGT=imageGT, imageET=[disparity_filled_v1])
            # error.evaluate()
            # print error.report
            # print

            #
            # # Improve Semi-global block matching using segmentation coded by skimage
            # t1 = time.time()
            # comp_map = segmentation.felzenszwalb(imageL, scale=100, sigma=0.5, min_size=50)
            #
            # dm = DisparityMap(imageLeft=imageL, imageRight=imageR)
            # disparity = dm.semiGlobalBlockMatch()
            #
            # disparity_filled_v2 = fill_occlusion(disparity=disparity, segmentation=comp_map)
            #
            # print 'Semi-global BM filled v2 saved, using time: ', time.time() - t1
            # filename = file[: -4] + '_BM_filled_v2.png'
            # # filename = 'disparity_BM_filled_v2.png'
            # cv2.imwrite(os.path.join(subdir, filename), disparity_filled_v2)
            #
            # error = EvaluationDM(imageGT=imageGT, imageET=[disparity_filled_v2])
            # error.evaluate()
            # print error.report
            # print
            #
            #
            # Improve Semi-global block matching using segmentation coded by skimage
            t1 = time.time()
            comp_map = segmentation.watershed(filters.sobel(imageL), markers=100, compactness=0.001)

            dm = DisparityMap(imageLeft=imageL, imageRight=imageR)
            disparity = dm.semiGlobalBlockMatch()

            disparity_filled_v3 = fill_occlusion(disparity=disparity, segmentation=comp_map)

            print 'Semi-global BM filled v3 saved, using time: ', time.time() - t1
            # filename = file[: -4] + '_BM_filled_v3.png'
            # filename = 'disparity_BM_filled_v3.png'
            # cv2.imwrite(os.path.join(subdir, filename), disparity_filled_v3)

            error = EvaluationDM(imageGT=imageGT, imageET=[disparity_filled_v3])
            error.evaluate()
            print error.report
            print

            # Improve Semi-global block matching using segmentation coded by skimage with half size image
            ratio = 2
            t1 = time.time()
            imageLs, imageRs, imageGTs = scaleDown(imageL, ratio), scaleDown(imageR, ratio), scaleDown(imageGT, ratio)

            dm = DisparityMap(imageLeft=imageLs, imageRight=imageRs)
            disparity = dm.semiGlobalBlockMatch()

            comp_map = segmentation.watershed(filters.sobel(imageLs), markers=100, compactness=0.001)
            disparity_filled_v3_f1 = fill_occlusion(disparity=disparity, segmentation=comp_map) * ratio

            print 'resize1: ', time.time() - t1
            filename = file[: -4] + '_BM_filled_v3_f1.png'
            # filename = 'disparity_BM_filled_v3_f1.png'
            cv2.imwrite(os.path.join(subdir, filename), disparity_filled_v3_f1)

            error = EvaluationDM(imageGT=imageGTs, imageET=[disparity_filled_v3_f1])
            error.evaluate()
            print error.report
            print

            # Improve Semi-global block matching using segmentation coded by skimage with half size image
            ratio = 2
            t1 = time.time()
            imageLs = cv2.resize(imageL, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_NEAREST)
            imageRs = cv2.resize(imageR, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_NEAREST)
            imageGTs = cv2.resize(imageGT, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_NEAREST)

            dm = DisparityMap(imageLeft=imageLs, imageRight=imageRs)
            disparity = dm.semiGlobalBlockMatch()

            comp_map = segmentation.watershed(filters.sobel(imageLs), markers=100, compactness=0.001)
            disparity_filled_v3_f2 = fill_occlusion(disparity=disparity, segmentation=comp_map) * ratio

            print 'resize2: ', time.time() - t1
            filename = file[: -4] + '_BM_filled_v3_f2.png'
            # filename = 'disparity_BM_filled_v3_f2.png'
            cv2.imwrite(os.path.join(subdir, filename), disparity_filled_v3_f2)

            error = EvaluationDM(imageGT=imageGTs, imageET=[disparity_filled_v3_f2])
            error.evaluate()
            print error.report
            print

            # Improve Semi-global block matching using segmentation coded by skimage with half size image
            ratio = 2
            t1 = time.time()
            imageLs = cv2.resize(imageL, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            imageRs = cv2.resize(imageR, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            imageGTs = cv2.resize(imageGT, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

            dm = DisparityMap(imageLeft=imageLs, imageRight=imageRs)
            disparity = dm.semiGlobalBlockMatch()

            comp_map = segmentation.watershed(filters.sobel(imageLs), markers=100, compactness=0.001)
            disparity_filled_v3_f3 = fill_occlusion(disparity=disparity, segmentation=comp_map) * ratio

            print 'resize3: ', time.time() - t1
            filename = file[: -4] + '_BM_filled_v3_f3.png'
            # filename = 'disparity_BM_filled_v3_f3.png'
            cv2.imwrite(os.path.join(subdir, filename), disparity_filled_v3_f3)

            error = EvaluationDM(imageGT=imageGTs, imageET=[disparity_filled_v3_f3])
            error.evaluate()
            print error.report
            print

            # Improve Semi-global block matching using segmentation coded by skimage with half size image
            ratio = 2
            t1 = time.time()
            imageLs = cv2.resize(imageL, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            imageRs = cv2.resize(imageR, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            imageGTs = cv2.resize(imageGT, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            dm = DisparityMap(imageLeft=imageLs, imageRight=imageRs)
            disparity = dm.semiGlobalBlockMatch()

            comp_map = segmentation.watershed(filters.sobel(imageLs), markers=100, compactness=0.001)
            disparity_filled_v3_f4 = fill_occlusion(disparity=disparity, segmentation=comp_map) * ratio

            print 'resize4: ', time.time() - t1
            filename = file[: -4] + '_BM_filled_v3_f4.png'
            # filename = 'disparity_BM_filled_v3_f4.png'
            cv2.imwrite(os.path.join(subdir, filename), disparity_filled_v3_f4)

            error = EvaluationDM(imageGT=imageGTs, imageET=[disparity_filled_v3_f4])
            error.evaluate()
            print error.report
            print

            # Improve Semi-global block matching using segmentation coded by skimage with half size image
            ratio = 2
            t1 = time.time()
            imageLs = cv2.resize(imageL, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            imageRs = cv2.resize(imageR, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            imageGTs = cv2.resize(imageGT, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

            dm = DisparityMap(imageLeft=imageLs, imageRight=imageRs)
            disparity = dm.semiGlobalBlockMatch()

            comp_map = segmentation.watershed(filters.sobel(imageLs), markers=100, compactness=0.001)
            disparity_filled_v3_f5 = fill_occlusion(disparity=disparity, segmentation=comp_map) * ratio

            print 'resize5: ', time.time() - t1
            filename = file[: -4] + '_BM_filled_v3_f5.png'
            # filename = 'disparity_BM_filled_v3_f5.png'
            cv2.imwrite(os.path.join(subdir, filename), disparity_filled_v3_f5)

            error = EvaluationDM(imageGT=imageGTs, imageET=[disparity_filled_v3_f5])
            error.evaluate()
            print error.report
            print

            # Improve Semi-global block matching using segmentation coded by skimage with half size image
            ratio = 2
            t1 = time.time()
            imageLs = cv2.resize(imageL, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LANCZOS4)
            imageRs = cv2.resize(imageR, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LANCZOS4)
            imageGTs = cv2.resize(imageGT, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LANCZOS4)

            dm = DisparityMap(imageLeft=imageLs, imageRight=imageRs)
            disparity = dm.semiGlobalBlockMatch()

            comp_map = segmentation.watershed(filters.sobel(imageLs), markers=100, compactness=0.001)
            disparity_filled_v3_f6 = fill_occlusion(disparity=disparity, segmentation=comp_map) * ratio

            print 'resize6: ', time.time() - t1
            filename = file[: -4] + '_BM_filled_v3_f6.png'
            # filename = 'disparity_BM_filled_v3_f6.png'
            cv2.imwrite(os.path.join(subdir, filename), disparity_filled_v3_f6)

            error = EvaluationDM(imageGT=imageGTs, imageET=[disparity_filled_v3_f6])
            error.evaluate()
            print error.report
            print


        print '---------------------------------------------------------------'



def improve_running_time():

    rootdir = 'stereo pair images/'

    # for subdir, dirs, files in os.walk(rootdir):
    #
    #     run, imageGT, imageL, imageR, imageL_c = False, None, None, None, None
    #     ratio = 1
    #     for f in files:
    #         filepath = os.path.join(subdir, f)
    #
    #         if filepath.endswith('disp0.pfm'):
    #             imageGT = readPFM(filepath)
    #             ratio = max(imageGT.shape) / 600
    #             imageGT = scaleDown(imageGT, ratio=ratio) * (1.0/ratio)
    #             cv2.imwrite(filepath[: -3] + 'png', imageGT)
    #             run = True
    #
    #         elif filepath.endswith('im0.png'):
    #             imageL = cv2.imread(filepath, 0)
    #             imageL = scaleDown(imageL, ratio=ratio)
    #
    #         elif filepath.endswith('im1.png'):
    #             imageR = cv2.imread(filepath, 0)
    #             imageR = scaleDown(imageR, ratio=ratio)


    for subdir, dirs, files in os.walk(rootdir):

        run, imageGT, imageL, imageR, imageL_c = False, None, None, None, None

        for f in files:
            filepath = os.path.join(subdir, f)

            if filepath.endswith('disp1.png'):
                imageGT = cv2.imread(filepath, 0) * 0.5
                # cv2.imwrite(os.path.join(subdir, 'disparity_GT.png'), imageGT)
                run = True

            elif filepath.endswith('view1.png'):
                imageL = cv2.imread(filepath, 0)

            elif filepath.endswith('view5.png'):
                imageR = cv2.imread(filepath, 0)


    # rootdir = 'data_scene_flow_12/training/'
    # rootdirGT    = rootdir + 'disp_occ'
    # rootdirLeft  = rootdir + 'image_0'
    # rootdirRight = rootdir + 'image_1'
    #
    # for file in os.listdir(rootdirLeft):
    #     run, subdir, imageGT, imageLeft, imageRight = False, None, None, None, None
    #     if file.endswith('_10.png'):
    #         imageGT   = cv2.imread(os.path.join(rootdirGT,    file), 0)
    #         imageL    = cv2.imread(os.path.join(rootdirLeft,  file), 0)
    #         imageR    = cv2.imread(os.path.join(rootdirRight, file), 0)
    #         run = True
    #         subdir = rootdir + 'output/'

        if run:
            print 'Building disparity map for ' + subdir + '...\n'
            cycle = 10

            total = 0
            for _ in range(cycle):
                temp = time.time()

                disparity = build_disparity(imageL=imageL, imageR=imageR, ndisparities=144)
                comp_map = build_segmentation(imageL=imageL)
                disparity_filled_v3 = fill_occlusion(disparity=disparity, segmentation=comp_map)

                total += time.time() - temp
            print 'Original running time: ', total / cycle


            total = 0
            for _ in range(cycle):
                temp = time.time()

                pool = ThreadPool(2)
                disparity = pool.apply_async(build_disparity, args=(imageL, imageR, 160))
                comp_map = pool.apply_async(build_segmentation, args=(imageL,))
                pool.close()
                pool.join()
                disparity_filled_v3 = fill_occlusion(disparity=disparity.get(), segmentation=comp_map.get())

                total += time.time() - temp
            print 'Multithread running time: ', total / cycle

            total = 0
            imageLs, imageRs = scaleDown(imageL, 2), scaleDown(imageR, 2)
            for _ in range(cycle):
                temp = time.time()

                disparity = build_disparity(imageL=imageLs, imageR=imageRs, ndisparities=80)
                comp_map = build_segmentation(imageL=imageLs)
                disparity_filled_v3 = fill_occlusion(disparity=disparity, segmentation=comp_map)

                total += time.time() - temp
            print 'Half size running time: ', total / cycle



        print '---------------------------------------------------------------'



if __name__ == '__main__':
    # improve_running_time()
    run()


'''
/usr/local/Cellar/python/2.7.13_1/Frameworks/Python.framework/Versions/2.7/bin/python2.7 /Users/hsiang-yunlee/LucidVR/main.py
---------------------------------------------------------------
Building disparity map for stereo pair images/Aloe...

Image info:  (555, 641) 112
Semi-global BM saved, using time:  1.86850380898
Bad matched pixel:    17.3903388568
Mean Absolute Error:  6.6619167

Image info:  (555, 641) 112
GC saved, using time:  81.5286428928
Bad matched pixel:    11.4896487751
Mean Absolute Error:  4.7621355

Image info:  (278, 321) 96
GC fast saved, using time:  14.6682422161
Bad matched pixel:    11.7277339742
Mean Absolute Error:  5.5991807

Semi-global BM filled saved, using time:  27.837594986
Bad matched pixel:    8.03558628832
Mean Absolute Error:  3.6847937

---------------------------------------------------------------
Building disparity map for stereo pair images/Art...

Image info:  (555, 695) 208
Semi-global BM saved, using time:  1.07463288307
Bad matched pixel:    30.1116080109
Mean Absolute Error:  18.055632

Image info:  (555, 695) 208
GC saved, using time:  204.219511032
Bad matched pixel:    24.1726618705
Mean Absolute Error:  14.033285

Image info:  (278, 348) 96
GC fast saved, using time:  21.0848989487
Bad matched pixel:    26.0278695962
Mean Absolute Error:  11.848734

Semi-global BM filled saved, using time:  32.4288790226
Bad matched pixel:    15.1094691814
Mean Absolute Error:  5.7225285

---------------------------------------------------------------
Building disparity map for stereo pair images/Baby1...

Image info:  (555, 620) 128
Semi-global BM saved, using time:  0.813862085342
Bad matched pixel:    12.3440860215
Mean Absolute Error:  4.532481

Image info:  (555, 620) 128
GC saved, using time:  112.298197985
Bad matched pixel:    4.33100842778
Mean Absolute Error:  1.8294376

Image info:  (278, 310) 48
GC fast saved, using time:  10.6693890095
Bad matched pixel:    5.71287416449
Mean Absolute Error:  2.186642

Semi-global BM filled saved, using time:  28.8627569675
Bad matched pixel:    3.62423714037
Mean Absolute Error:  1.8691922

---------------------------------------------------------------
Building disparity map for stereo pair images/Baby2...

Image info:  (555, 620) 128
Semi-global BM saved, using time:  0.726569890976
Bad matched pixel:    11.3836094159
Mean Absolute Error:  4.6462226

Image info:  (555, 620) 128
GC saved, using time:  123.397757053
Bad matched pixel:    5.30921243824
Mean Absolute Error:  2.795427

Image info:  (278, 310) 48
GC fast saved, using time:  10.9164919853
Bad matched pixel:    5.41586748038
Mean Absolute Error:  3.043605

Semi-global BM filled saved, using time:  26.6916439533
Bad matched pixel:    5.547224644
Mean Absolute Error:  2.901127

---------------------------------------------------------------
Building disparity map for stereo pair images/Baby3...

Image info:  (555, 656) 144
Semi-global BM saved, using time:  0.758260011673
Bad matched pixel:    15.8476159086
Mean Absolute Error:  9.781199

Image info:  (555, 656) 144
GC saved, using time:  134.48516202
Bad matched pixel:    21.2332454406
Mean Absolute Error:  13.240475

Image info:  (278, 328) 64
GC fast saved, using time:  13.9548480511
Bad matched pixel:    20.2417051198
Mean Absolute Error:  13.0336685

Semi-global BM filled saved, using time:  29.7566380501
Bad matched pixel:    9.10019775873
Mean Absolute Error:  4.9474435

---------------------------------------------------------------
Building disparity map for stereo pair images/Books...

Image info:  (555, 695) 208
Semi-global BM saved, using time:  1.10425496101
Bad matched pixel:    14.8302547151
Mean Absolute Error:  10.307922

Image info:  (555, 695) 208
GC saved, using time:  160.038744926
Bad matched pixel:    9.45647806079
Mean Absolute Error:  4.5637164

Image info:  (278, 348) 96
GC fast saved, using time:  18.5404760838
Bad matched pixel:    8.90712295029
Mean Absolute Error:  3.903785

Semi-global BM filled saved, using time:  30.0567238331
Bad matched pixel:    7.87348499579
Mean Absolute Error:  4.0198436

---------------------------------------------------------------
Building disparity map for stereo pair images/Bowling1...

Image info:  (555, 626) 160
Semi-global BM saved, using time:  0.720799922943
Bad matched pixel:    21.2177992689
Mean Absolute Error:  16.49186

Image info:  (555, 626) 160
GC saved, using time:  458.405312061
Bad matched pixel:    31.2557925337
Mean Absolute Error:  25.077673

Image info:  (278, 313) 80
GC fast saved, using time:  44.3653140068
Bad matched pixel:    33.6985867657
Mean Absolute Error:  11.516273

Semi-global BM filled saved, using time:  28.6803860664
Bad matched pixel:    10.8980226233
Mean Absolute Error:  6.6934104

---------------------------------------------------------------
Building disparity map for stereo pair images/Bowling2...

Image info:  (555, 665) 192
Semi-global BM saved, using time:  0.96443605423
Bad matched pixel:    19.6350335298
Mean Absolute Error:  12.524624

Image info:  (555, 665) 192
GC saved, using time:  219.005023003
Bad matched pixel:    16.1501049922
Mean Absolute Error:  9.516072

Image info:  (278, 333) 96
GC fast saved, using time:  17.1516120434
Bad matched pixel:    10.7566212829
Mean Absolute Error:  6.498797

Semi-global BM filled saved, using time:  30.8001480103
Bad matched pixel:    9.56336787916
Mean Absolute Error:  6.1361837

---------------------------------------------------------------
Building disparity map for stereo pair images/Cloth1...

Image info:  (555, 626) 160
Semi-global BM saved, using time:  2.73974990845
Bad matched pixel:    10.3704343321
Mean Absolute Error:  7.393419

Image info:  (555, 626) 160
GC saved, using time:  95.9358079433
Bad matched pixel:    2.51273637855
Mean Absolute Error:  0.96751434

Image info:  (278, 313) 80
GC fast saved, using time:  9.11187982559
Bad matched pixel:    1.53642460352
Mean Absolute Error:  1.0503829

Semi-global BM filled saved, using time:  26.6064360142
Bad matched pixel:    2.47128918055
Mean Absolute Error:  2.6013973

---------------------------------------------------------------
Building disparity map for stereo pair images/Cloth2...

Image info:  (555, 650) 224
Semi-global BM saved, using time:  1.18042206764
Bad matched pixel:    15.5595287595
Mean Absolute Error:  11.171296

Image info:  (555, 650) 224
GC saved, using time:  157.172151089
Bad matched pixel:    5.10713790714
Mean Absolute Error:  3.5768344

Image info:  (278, 325) 96
GC fast saved, using time:  17.3892719746
Bad matched pixel:    6.70547470547
Mean Absolute Error:  3.7487707

Semi-global BM filled saved, using time:  27.4910049438
Bad matched pixel:    6.41053361053
Mean Absolute Error:  4.2885623

---------------------------------------------------------------
Building disparity map for stereo pair images/Cloth3...

Image info:  (555, 626) 160
Semi-global BM saved, using time:  1.08603310585
Bad matched pixel:    11.5208819043
Mean Absolute Error:  6.8910637

Image info:  (555, 626) 160
GC saved, using time:  93.5416858196
Bad matched pixel:    3.06018478542
Mean Absolute Error:  1.7100021

Image info:  (278, 313) 80
GC fast saved, using time:  9.61662101746
Bad matched pixel:    2.84805572346
Mean Absolute Error:  1.9446911

Semi-global BM filled saved, using time:  26.2877140045
Bad matched pixel:    3.64044555738
Mean Absolute Error:  2.6593788

---------------------------------------------------------------
Building disparity map for stereo pair images/Cloth4...

Image info:  (555, 650) 192
Semi-global BM saved, using time:  1.73875784874
Bad matched pixel:    16.842966043
Mean Absolute Error:  11.324147

Image info:  (555, 650) 192
GC saved, using time:  131.284832954
Bad matched pixel:    7.34968814969
Mean Absolute Error:  5.1183395

Image info:  (278, 325) 96
GC fast saved, using time:  14.2951939106
Bad matched pixel:    7.05613305613
Mean Absolute Error:  4.6852393

Semi-global BM filled saved, using time:  27.7242078781
Bad matched pixel:    7.56063756064
Mean Absolute Error:  5.4485955

---------------------------------------------------------------
---------------------------------------------------------------
Building disparity map for stereo pair images/Dolls...

Image info:  (555, 695) 304
Semi-global BM saved, using time:  1.58257198334
Bad matched pixel:    14.4208957159
Mean Absolute Error:  10.72352

Image info:  (555, 695) 304
GC saved, using time:  200.917653799
Bad matched pixel:    6.45485773543
Mean Absolute Error:  5.396587

Image info:  (278, 348) 96
GC fast saved, using time:  16.1408808231
Bad matched pixel:    5.64858383563
Mean Absolute Error:  4.132453

Semi-global BM filled saved, using time:  34.2794969082
Bad matched pixel:    6.2031239873
Mean Absolute Error:  3.6939712

---------------------------------------------------------------
---------------------------------------------------------------
---------------------------------------------------------------
Building disparity map for stereo pair images/Flowerpots...

Image info:  (555, 656) 176
Semi-global BM saved, using time:  0.827574968338
Bad matched pixel:    20.9236980883
Mean Absolute Error:  13.770644

Image info:  (555, 656) 176
GC saved, using time:  165.742943048
Bad matched pixel:    17.3242144584
Mean Absolute Error:  11.87812

Image info:  (278, 328) 80
GC fast saved, using time:  14.8500831127
Bad matched pixel:    17.3082838936
Mean Absolute Error:  12.353126

Semi-global BM filled saved, using time:  30.9352779388
Bad matched pixel:    17.7065480114
Mean Absolute Error:  10.878958

---------------------------------------------------------------
Building disparity map for stereo pair images/Lampshade1...

Image info:  (555, 650) 144
Semi-global BM saved, using time:  0.705530881882
Bad matched pixel:    22.0340956341
Mean Absolute Error:  10.436874

Image info:  (555, 650) 144
GC saved, using time:  233.799348116
Bad matched pixel:    15.3460845461
Mean Absolute Error:  5.7650175

Image info:  (278, 325) 64
GC fast saved, using time:  14.9512240887
Bad matched pixel:    17.7582813583
Mean Absolute Error:  6.87834

Semi-global BM filled saved, using time:  32.1910829544
Bad matched pixel:    9.55426195426
Mean Absolute Error:  4.6839333

---------------------------------------------------------------
Building disparity map for stereo pair images/Lampshade2...

Image info:  (555, 650) 112
Semi-global BM saved, using time:  0.662575006485
Bad matched pixel:    23.8297990298
Mean Absolute Error:  12.383896

Image info:  (555, 650) 112
GC saved, using time:  213.088649035
Bad matched pixel:    22.195980596
Mean Absolute Error:  8.137437

Image info:  (278, 325) 64
GC fast saved, using time:  26.2410809994
Bad matched pixel:    27.2729036729
Mean Absolute Error:  9.274862

Semi-global BM filled saved, using time:  30.3180828094
Bad matched pixel:    6.18794178794
Mean Absolute Error:  3.3021212

---------------------------------------------------------------
Building disparity map for stereo pair images/Laundry...

Image info:  (555, 671) 432
Semi-global BM saved, using time:  2.04439496994
Bad matched pixel:    22.9916354507
Mean Absolute Error:  15.631862

Image info:  (555, 671) 432
GC saved, using time:  278.549813032
Bad matched pixel:    18.1106590943
Mean Absolute Error:  10.009376

Image info:  (278, 336) 224
GC fast saved, using time:  29.413613081
Bad matched pixel:    16.8118043528
Mean Absolute Error:  8.278577

Semi-global BM filled saved, using time:  29.5040259361
Bad matched pixel:    9.70663659188
Mean Absolute Error:  5.259488

---------------------------------------------------------------
Building disparity map for stereo pair images/Midd1...

Image info:  (555, 698) 160
Semi-global BM saved, using time:  0.938793897629
Bad matched pixel:    47.6142905083
Mean Absolute Error:  17.230614

Image info:  (555, 698) 160
GC saved, using time:  602.325881958
Bad matched pixel:    48.8316683446
Mean Absolute Error:  13.653513

Image info:  (278, 349) 80
GC fast saved, using time:  50.3050689697
Bad matched pixel:    47.5311701386
Mean Absolute Error:  14.4882145

Semi-global BM filled saved, using time:  33.8853960037
Bad matched pixel:    33.0790676063
Mean Absolute Error:  9.141466

---------------------------------------------------------------
Building disparity map for stereo pair images/Midd2...

Image info:  (555, 683) 208
Semi-global BM saved, using time:  1.02530789375
Bad matched pixel:    45.8937121602
Mean Absolute Error:  12.26307

Image info:  (555, 683) 208
GC saved, using time:  532.364686012
Bad matched pixel:    23.2366480683
Mean Absolute Error:  7.0937595

Image info:  (278, 342) 80
GC fast saved, using time:  111.233355999
Bad matched pixel:    51.7744977774
Mean Absolute Error:  12.322174

Semi-global BM filled saved, using time:  30.6576359272
Bad matched pixel:    41.5968237637
Mean Absolute Error:  10.103815

---------------------------------------------------------------
Building disparity map for stereo pair images/Moebius...

Image info:  (555, 695) 208
Semi-global BM saved, using time:  1.11086893082
Bad matched pixel:    18.1590511375
Mean Absolute Error:  10.9249

Image info:  (555, 695) 208
GC saved, using time:  182.098573923
Bad matched pixel:    11.4247196837
Mean Absolute Error:  6.055467

Image info:  (278, 348) 80
GC fast saved, using time:  15.2152121067
Bad matched pixel:    11.4335342537
Mean Absolute Error:  5.703029

Semi-global BM filled saved, using time:  31.9151210785
Bad matched pixel:    7.72078553374
Mean Absolute Error:  3.3378222

---------------------------------------------------------------
Building disparity map for stereo pair images/Monopoly...

Image info:  (555, 665) 144
Semi-global BM saved, using time:  0.843655824661
Bad matched pixel:    11.5250287882
Mean Absolute Error:  7.700632

Image info:  (555, 665) 144
GC saved, using time:  209.922646046
Bad matched pixel:    37.9196640249
Mean Absolute Error:  17.326004

Image info:  (278, 333) 48
GC fast saved, using time:  12.8315739632
Bad matched pixel:    41.4129919393
Mean Absolute Error:  17.892517

Semi-global BM filled saved, using time:  30.1337471008
Bad matched pixel:    24.7532344374
Mean Absolute Error:  12.201301

---------------------------------------------------------------
Building disparity map for stereo pair images/Plastic...

Image info:  (555, 635) 144
Semi-global BM saved, using time:  0.747352838516
Bad matched pixel:    36.4505923246
Mean Absolute Error:  24.632143

Image info:  (555, 635) 144
GC saved, using time:  4892.90919805
Bad matched pixel:    26.3718521671
Mean Absolute Error:  13.2373905

Image info:  (278, 318) 64
GC fast saved, using time:  26.6926081181
Bad matched pixel:    27.9764488898
Mean Absolute Error:  11.494086

Semi-global BM filled saved, using time:  26.2794728279
Bad matched pixel:    13.8670639143
Mean Absolute Error:  4.910908

---------------------------------------------------------------
Building disparity map for stereo pair images/Reindeer...

Image info:  (555, 671) 192
Semi-global BM saved, using time:  0.88719701767
Bad matched pixel:    22.1882090735
Mean Absolute Error:  12.705008

Image info:  (555, 671) 192
GC saved, using time:  250.551471949
Bad matched pixel:    13.9804782428
Mean Absolute Error:  6.8554945

Image info:  (278, 336) 96
GC fast saved, using time:  24.723389864
Bad matched pixel:    18.4012029914
Mean Absolute Error:  6.823825

Semi-global BM filled saved, using time:  27.5922009945
Bad matched pixel:    8.38737396114
Mean Absolute Error:  4.271708

---------------------------------------------------------------
Building disparity map for stereo pair images/Rocks1...

Image info:  (555, 638) 160
Semi-global BM saved, using time:  0.896329879761
Bad matched pixel:    13.3680702646
Mean Absolute Error:  8.839084

Image info:  (555, 638) 160
GC saved, using time:  99.2664442062
Bad matched pixel:    11.1392583806
Mean Absolute Error:  6.7941117

Image info:  (278, 319) 64
GC fast saved, using time:  9.56203484535
Bad matched pixel:    10.3660086419
Mean Absolute Error:  6.0969357

Semi-global BM filled saved, using time:  28.3065989017
Bad matched pixel:    7.69917252676
Mean Absolute Error:  4.939689

---------------------------------------------------------------
Building disparity map for stereo pair images/Rocks2...

Image info:  (555, 638) 160
Semi-global BM saved, using time:  0.899432897568
Bad matched pixel:    14.6053263295
Mean Absolute Error:  9.7946005

Image info:  (555, 638) 160
GC saved, using time:  91.313434124
Bad matched pixel:    12.9986726538
Mean Absolute Error:  8.3961935

Image info:  (278, 319) 80
GC fast saved, using time:  10.1242380142
Bad matched pixel:    10.8955350335
Mean Absolute Error:  7.1577764

Semi-global BM filled saved, using time:  26.6311528683
Bad matched pixel:    6.93580728063
Mean Absolute Error:  4.614797

---------------------------------------------------------------
Building disparity map for stereo pair images/Wood1...

Image info:  (555, 686) 192
Semi-global BM saved, using time:  0.79469704628
Bad matched pixel:    17.1612428755
Mean Absolute Error:  10.584671

Image info:  (555, 686) 192
GC saved, using time:  206.465393066
Bad matched pixel:    5.71481102093
Mean Absolute Error:  2.8289208

Image info:  (278, 343) 96
GC fast saved, using time:  20.9750390053
Bad matched pixel:    7.34168570903
Mean Absolute Error:  3.0604892

Semi-global BM filled saved, using time:  29.1203410625
Bad matched pixel:    4.97097680771
Mean Absolute Error:  3.151188

---------------------------------------------------------------
Building disparity map for stereo pair images/Wood2...

Image info:  (555, 653) 368
Semi-global BM saved, using time:  1.38682508469
Bad matched pixel:    16.2807830802
Mean Absolute Error:  12.349345

Image info:  (555, 653) 368
GC saved, using time:  549.157200098
Bad matched pixel:    18.3596153581
Mean Absolute Error:  7.704993

Image info:  (278, 327) 64
GC fast saved, using time:  34.2676169872
Bad matched pixel:    12.1747168302
Mean Absolute Error:  4.2271485

Semi-global BM filled saved, using time:  28.2749340534
Bad matched pixel:    6.21800974021
Mean Absolute Error:  4.783492

---------------------------------------------------------------

'''

'''
/usr/local/Cellar/python/2.7.13_1/Frameworks/Python.framework/Versions/2.7/bin/python2.7 /Users/hsiang-yunlee/LucidVR/main.py
---------------------------------------------------------------
Building disparity map for stereo pair images/Aloe...

Image info:  (555, 641) 112
Semi-global BM saved, using time:  1.72113394737
Bad matched pixel:    18.2412053239
Mean Absolute Error:  6.66191673343

Image info:  (278, 321) 96
GC fast saved, using time:  14.3141598701
Bad matched pixel:    17.680426136
Mean Absolute Error:  5.5991807

Semi-global BM filled saved, using time:  26.4382209778
Bad matched pixel:    10.3793340923
Mean Absolute Error:  3.68479360796

Semi-global BM filled v1 saved, using time:  0.582515954971
Bad matched pixel:    10.5308428553
Mean Absolute Error:  3.65789482931

---------------------------------------------------------------
Building disparity map for stereo pair images/Art...

Image info:  (555, 695) 208
Semi-global BM saved, using time:  1.01114583015
Bad matched pixel:    31.8727072396
Mean Absolute Error:  18.0556314408

Image info:  (278, 348) 96
GC fast saved, using time:  20.2017250061
Bad matched pixel:    37.4554410526
Mean Absolute Error:  11.848734

Semi-global BM filled saved, using time:  28.6622400284
Bad matched pixel:    20.9517143042
Mean Absolute Error:  5.72252851773

Semi-global BM filled v1 saved, using time:  0.875408887863
Bad matched pixel:    19.632380582
Mean Absolute Error:  6.55651224966

---------------------------------------------------------------
Building disparity map for stereo pair images/Baby1...

Image info:  (555, 620) 128
Semi-global BM saved, using time:  0.773440122604
Bad matched pixel:    12.6393490264
Mean Absolute Error:  4.53248074688

Image info:  (278, 310) 48
GC fast saved, using time:  10.4388859272
Bad matched pixel:    9.64370822435
Mean Absolute Error:  2.186642

Semi-global BM filled saved, using time:  25.3343250751
Bad matched pixel:    4.53211275792
Mean Absolute Error:  1.86919227695

Semi-global BM filled v1 saved, using time:  0.555606126785
Bad matched pixel:    4.56030223772
Mean Absolute Error:  1.93709223336

---------------------------------------------------------------
Building disparity map for stereo pair images/Baby2...

Image info:  (555, 620) 128
Semi-global BM saved, using time:  0.729250907898
Bad matched pixel:    12.1490845684
Mean Absolute Error:  4.64622366318

Image info:  (278, 310) 48
GC fast saved, using time:  10.7018618584
Bad matched pixel:    17.8634117989
Mean Absolute Error:  3.043605

Semi-global BM filled saved, using time:  25.8133900166
Bad matched pixel:    7.31793083406
Mean Absolute Error:  2.90112721593

Semi-global BM filled v1 saved, using time:  0.644604921341
Bad matched pixel:    6.79453647196
Mean Absolute Error:  2.56156894798

---------------------------------------------------------------
Building disparity map for stereo pair images/Baby3...

Image info:  (555, 656) 144
Semi-global BM saved, using time:  0.689306020737
Bad matched pixel:    17.069325423
Mean Absolute Error:  9.78120004532

Image info:  (278, 328) 64
GC fast saved, using time:  12.4935817719
Bad matched pixel:    32.2819160624
Mean Absolute Error:  13.0336685

Semi-global BM filled saved, using time:  26.784965992
Bad matched pixel:    11.3321248077
Mean Absolute Error:  4.9474428697

Semi-global BM filled v1 saved, using time:  0.610315084457
Bad matched pixel:    12.1558448693
Mean Absolute Error:  5.43798650022

---------------------------------------------------------------
Building disparity map for stereo pair images/Books...

Image info:  (555, 695) 208
Semi-global BM saved, using time:  1.01553106308
Bad matched pixel:    17.9428349213
Mean Absolute Error:  10.3079216087

Image info:  (278, 348) 96
GC fast saved, using time:  17.7614860535
Bad matched pixel:    19.8473005379
Mean Absolute Error:  3.903785

Semi-global BM filled saved, using time:  29.296900034
Bad matched pixel:    13.0694147385
Mean Absolute Error:  4.01984380064

Semi-global BM filled v1 saved, using time:  0.68695306778
Bad matched pixel:    13.4121459589
Mean Absolute Error:  3.77230799144

---------------------------------------------------------------
Building disparity map for stereo pair images/Bowling1...

Image info:  (555, 626) 160
Semi-global BM saved, using time:  0.686874866486
Bad matched pixel:    26.316380278
Mean Absolute Error:  16.4918595113

Image info:  (278, 313) 80
GC fast saved, using time:  41.7824678421
Bad matched pixel:    50.3361828282
Mean Absolute Error:  11.516273

Semi-global BM filled saved, using time:  25.1796529293
Bad matched pixel:    16.3330742883
Mean Absolute Error:  6.6934089241

Semi-global BM filled v1 saved, using time:  0.680341959
Bad matched pixel:    17.514607259
Mean Absolute Error:  7.6064216173

---------------------------------------------------------------
Building disparity map for stereo pair images/Bowling2...

Image info:  (555, 665) 192
Semi-global BM saved, using time:  0.8326420784
Bad matched pixel:    20.7830386778
Mean Absolute Error:  12.5246223667

Image info:  (278, 333) 96
GC fast saved, using time:  15.3338420391
Bad matched pixel:    32.619115356
Mean Absolute Error:  6.498797

Semi-global BM filled saved, using time:  27.3543901443
Bad matched pixel:    13.2433787171
Mean Absolute Error:  6.13618285579

Semi-global BM filled v1 saved, using time:  0.67471909523
Bad matched pixel:    13.9058456953
Mean Absolute Error:  6.07415870758

---------------------------------------------------------------
Building disparity map for stereo pair images/Cloth1...

Image info:  (555, 626) 160
Semi-global BM saved, using time:  2.50883507729
Bad matched pixel:    10.4340442679
Mean Absolute Error:  7.39341899807

Image info:  (278, 313) 80
GC fast saved, using time:  9.16867017746
Bad matched pixel:    4.09809170193
Mean Absolute Error:  1.0503829

Semi-global BM filled saved, using time:  25.7933058739
Bad matched pixel:    4.79434706272
Mean Absolute Error:  2.6013972239

Semi-global BM filled v1 saved, using time:  0.486867904663
Bad matched pixel:    8.13027084593
Mean Absolute Error:  2.47057644705

---------------------------------------------------------------
Building disparity map for stereo pair images/Cloth2...

Image info:  (555, 650) 208
Semi-global BM saved, using time:  1.07850003242
Bad matched pixel:    16.4734580735
Mean Absolute Error:  11.1699630977

Image info:  (278, 325) 96
GC fast saved, using time:  17.1492779255
Bad matched pixel:    15.7316701317
Mean Absolute Error:  3.7487707

Semi-global BM filled saved, using time:  26.5147740841
Bad matched pixel:    11.1254331254
Mean Absolute Error:  4.28816164241

Semi-global BM filled v1 saved, using time:  0.587307929993
Bad matched pixel:    11.0938322938
Mean Absolute Error:  5.27412629938

---------------------------------------------------------------
Building disparity map for stereo pair images/Cloth3...

Image info:  (555, 626) 160
Semi-global BM saved, using time:  1.01952600479
Bad matched pixel:    12.557349682
Mean Absolute Error:  6.8910636675

Image info:  (278, 313) 80
GC fast saved, using time:  9.53382110596
Bad matched pixel:    6.6911320266
Mean Absolute Error:  1.9446911

Semi-global BM filled saved, using time:  25.4170072079
Bad matched pixel:    6.71559738652
Mean Absolute Error:  2.65937886769

Semi-global BM filled v1 saved, using time:  0.571691989899
Bad matched pixel:    7.60843911004
Mean Absolute Error:  3.39262981032

---------------------------------------------------------------
Building disparity map for stereo pair images/Cloth4...

Image info:  (555, 650) 192
Semi-global BM saved, using time:  1.69245100021
Bad matched pixel:    17.511018711
Mean Absolute Error:  11.3241470894

Image info:  (278, 325) 96
GC fast saved, using time:  13.7063140869
Bad matched pixel:    13.3527373527
Mean Absolute Error:  4.6852393

Semi-global BM filled saved, using time:  26.51217103
Bad matched pixel:    9.81732501733
Mean Absolute Error:  5.44859476784

Semi-global BM filled v1 saved, using time:  0.603557109833
Bad matched pixel:    10.1178101178
Mean Absolute Error:  6.48291025641

---------------------------------------------------------------
Building disparity map for stereo pair images/Dolls...

Image info:  (555, 695) 224
Semi-global BM saved, using time:  1.18729400635
Bad matched pixel:    18.0698684296
Mean Absolute Error:  10.6692855985

Image info:  (278, 348) 96
GC fast saved, using time:  13.5137329102
Bad matched pixel:    18.9062155681
Mean Absolute Error:  4.132453

Semi-global BM filled saved, using time:  28.0699560642
Bad matched pixel:    13.0250826366
Mean Absolute Error:  3.6771135524

Semi-global BM filled v1 saved, using time:  0.722463130951
Bad matched pixel:    12.8062738998
Mean Absolute Error:  3.78514275066

---------------------------------------------------------------
Building disparity map for stereo pair images/Flowerpots...

Image info:  (555, 656) 176
Semi-global BM saved, using time:  0.735754013062
Bad matched pixel:    22.1769940672
Mean Absolute Error:  13.7706463895

Image info:  (278, 328) 80
GC fast saved, using time:  13.6353728771
Bad matched pixel:    57.3143265216
Mean Absolute Error:  12.353126

Semi-global BM filled saved, using time:  26.29946208
Bad matched pixel:    21.9102944408
Mean Absolute Error:  10.8789579213

Semi-global BM filled v1 saved, using time:  0.714380025864
Bad matched pixel:    24.7659854977
Mean Absolute Error:  13.0246491156

---------------------------------------------------------------
Building disparity map for stereo pair images/Lampshade1...

Image info:  (555, 650) 144
Semi-global BM saved, using time:  0.606876850128
Bad matched pixel:    22.8318780319
Mean Absolute Error:  10.4368764726

Image info:  (278, 325) 64
GC fast saved, using time:  13.8789069653
Bad matched pixel:    42.9402633403
Mean Absolute Error:  6.87834

Semi-global BM filled saved, using time:  27.0331540108
Bad matched pixel:    12.0127512128
Mean Absolute Error:  4.68393399168

Semi-global BM filled v1 saved, using time:  0.70049405098
Bad matched pixel:    13.9537075537
Mean Absolute Error:  5.84584251559

---------------------------------------------------------------
Building disparity map for stereo pair images/Lampshade2...

Image info:  (555, 650) 112
Semi-global BM saved, using time:  0.550747156143
Bad matched pixel:    24.6264726265
Mean Absolute Error:  12.3838970894

Image info:  (278, 325) 64
GC fast saved, using time:  22.7532439232
Bad matched pixel:    57.7161469161
Mean Absolute Error:  9.274862

Semi-global BM filled saved, using time:  26.1653540134
Bad matched pixel:    8.45654885655
Mean Absolute Error:  3.30212058212

Semi-global BM filled v1 saved, using time:  0.676523923874
Bad matched pixel:    13.3341649342
Mean Absolute Error:  5.53819040194

---------------------------------------------------------------
Building disparity map for stereo pair images/Laundry...

Image info:  (555, 671) 208
Semi-global BM saved, using time:  0.975404977798
Bad matched pixel:    28.9598689599
Mean Absolute Error:  15.1964420456

Image info:  (278, 336) 112
GC fast saved, using time:  17.2559430599
Bad matched pixel:    29.3806474134
Mean Absolute Error:  6.733858

Semi-global BM filled saved, using time:  26.655148983
Bad matched pixel:    22.1710234825
Mean Absolute Error:  5.72064016326

Semi-global BM filled v1 saved, using time:  0.670521020889
Bad matched pixel:    24.8353271304
Mean Absolute Error:  8.11167965656

---------------------------------------------------------------
Building disparity map for stereo pair images/Midd1...

Image info:  (555, 698) 160
Semi-global BM saved, using time:  0.81769990921
Bad matched pixel:    49.3967319755
Mean Absolute Error:  17.2306117543

Image info:  (278, 349) 80
GC fast saved, using time:  44.1130111217
Bad matched pixel:    53.9131108186
Mean Absolute Error:  14.4882145

Semi-global BM filled saved, using time:  27.8381791115
Bad matched pixel:    38.2841580836
Mean Absolute Error:  9.14146644854

Semi-global BM filled v1 saved, using time:  0.918990135193
Bad matched pixel:    43.6931774181
Mean Absolute Error:  11.0662289618

---------------------------------------------------------------
Building disparity map for stereo pair images/Midd2...

Image info:  (555, 683) 208
Semi-global BM saved, using time:  0.921580076218
Bad matched pixel:    46.9399707174
Mean Absolute Error:  12.2630685041

Image info:  (278, 342) 80
GC fast saved, using time:  92.8371539116
Bad matched pixel:    61.088995291
Mean Absolute Error:  12.322174

Semi-global BM filled saved, using time:  27.4408159256
Bad matched pixel:    44.8099930091
Mean Absolute Error:  10.1038146492

Semi-global BM filled v1 saved, using time:  0.87393283844
Bad matched pixel:    41.6730639864
Mean Absolute Error:  9.73263836677

---------------------------------------------------------------
Building disparity map for stereo pair images/Moebius...

Image info:  (555, 695) 208
Semi-global BM saved, using time:  1.01731491089
Bad matched pixel:    21.7769136043
Mean Absolute Error:  10.9248990537

Image info:  (278, 348) 80
GC fast saved, using time:  14.4570670128
Bad matched pixel:    22.5035971223
Mean Absolute Error:  5.703029

Semi-global BM filled saved, using time:  28.8979170322
Bad matched pixel:    13.8800959233
Mean Absolute Error:  3.33782244475

Semi-global BM filled v1 saved, using time:  0.714520931244
Bad matched pixel:    14.2536781386
Mean Absolute Error:  4.10095696416

---------------------------------------------------------------
Building disparity map for stereo pair images/Monopoly...

Image info:  (555, 665) 144
Semi-global BM saved, using time:  0.744281053543
Bad matched pixel:    36.8732642417
Mean Absolute Error:  7.700631308

Image info:  (278, 333) 48
GC fast saved, using time:  11.8483259678
Bad matched pixel:    52.4971889182
Mean Absolute Error:  17.892517

Semi-global BM filled saved, using time:  26.319355011
Bad matched pixel:    31.6894940053
Mean Absolute Error:  12.2013029195

Semi-global BM filled v1 saved, using time:  0.769673109055
Bad matched pixel:    32.4186141028
Mean Absolute Error:  9.24766731017

---------------------------------------------------------------
Building disparity map for stereo pair images/Plastic...

Image info:  (555, 635) 144
Semi-global BM saved, using time:  0.608820199966
Bad matched pixel:    39.5573526282
Mean Absolute Error:  24.6321454919

Image info:  (278, 318) 64
GC fast saved, using time:  23.7407021523
Bad matched pixel:    49.6871674824
Mean Absolute Error:  11.494086

Semi-global BM filled saved, using time:  24.8324210644
Bad matched pixel:    19.8422359367
Mean Absolute Error:  4.91090905866

Semi-global BM filled v1 saved, using time:  0.824400901794
Bad matched pixel:    19.832588494
Mean Absolute Error:  5.75633166631

---------------------------------------------------------------
Building disparity map for stereo pair images/Reindeer...

Image info:  (555, 671) 192
Semi-global BM saved, using time:  0.895900011063
Bad matched pixel:    23.6642902217
Mean Absolute Error:  12.7050078208

Image info:  (278, 336) 96
GC fast saved, using time:  24.2934339046
Bad matched pixel:    36.4302305286
Mean Absolute Error:  6.823825

Semi-global BM filled saved, using time:  26.9230649471
Bad matched pixel:    12.6609470872
Mean Absolute Error:  4.27170906003

Semi-global BM filled v1 saved, using time:  0.66958117485
Bad matched pixel:    15.8131604033
Mean Absolute Error:  5.55954404479

---------------------------------------------------------------
Building disparity map for stereo pair images/Rocks1...

Image info:  (555, 638) 160
Semi-global BM saved, using time:  0.916676998138
Bad matched pixel:    14.4675082606
Mean Absolute Error:  8.83908469598

Image info:  (278, 319) 64
GC fast saved, using time:  9.89321303368
Bad matched pixel:    13.6778785055
Mean Absolute Error:  6.0969357

Semi-global BM filled saved, using time:  25.8821971416
Bad matched pixel:    10.2482419724
Mean Absolute Error:  4.93968976814

Semi-global BM filled v1 saved, using time:  0.563169002533
Bad matched pixel:    12.0955124403
Mean Absolute Error:  5.21321468412

---------------------------------------------------------------
Building disparity map for stereo pair images/Rocks2...

Image info:  (555, 638) 160
Semi-global BM saved, using time:  0.961336135864
Bad matched pixel:    15.8256375498
Mean Absolute Error:  9.79460006637

Image info:  (278, 319) 80
GC fast saved, using time:  10.1356151104
Bad matched pixel:    17.0674122398
Mean Absolute Error:  7.1577764

Semi-global BM filled saved, using time:  25.8890800476
Bad matched pixel:    9.9296788952
Mean Absolute Error:  4.61479708549

Semi-global BM filled v1 saved, using time:  0.570420980453
Bad matched pixel:    10.4414132
Mean Absolute Error:  4.49641298822

---------------------------------------------------------------
Building disparity map for stereo pair images/Wood1...

Image info:  (555, 686) 192
Semi-global BM saved, using time:  0.775249004364
Bad matched pixel:    17.5770230872
Mean Absolute Error:  10.5846712237

Image info:  (278, 343) 96
GC fast saved, using time:  20.7593519688
Bad matched pixel:    20.6062038715
Mean Absolute Error:  3.0604892

Semi-global BM filled saved, using time:  28.3622541428
Bad matched pixel:    8.02195781788
Mean Absolute Error:  3.15118702887

Semi-global BM filled v1 saved, using time:  0.615957975388
Bad matched pixel:    10.6913035484
Mean Absolute Error:  4.39480685131

---------------------------------------------------------------
Building disparity map for stereo pair images/Wood2...

Image info:  (555, 653) 208
Semi-global BM saved, using time:  0.806618213654
Bad matched pixel:    16.380116717
Mean Absolute Error:  12.3496564712

Image info:  (278, 327) 64
GC fast saved, using time:  30.9491438866
Bad matched pixel:    35.4615565029
Mean Absolute Error:  4.2271485

Semi-global BM filled saved, using time:  27.2808759212
Bad matched pixel:    7.62275292137
Mean Absolute Error:  4.78373170261

Semi-global BM filled v1 saved, using time:  0.596494913101
Bad matched pixel:    8.78964722763
Mean Absolute Error:  4.96483451292

---------------------------------------------------------------

Process finished with exit code 0

'''

'''
---------------------------------------------------------------
Building disparity map for stereo pair images2/Jadeplant-perfect...

Image info:  (497, 658) 160
Semi-global BM saved, using time:  0.919097900391
Bad matched pixel:    80.3
Mean Absolute Error:  45.9

Image info:  (497, 658) 160
GC saved, using time:  181.756631851
Bad matched pixel:    80.3
Mean Absolute Error:  45.9

Image info:  (249, 329) 80
GC fast saved, using time:  20.6702539921
Bad matched pixel:    84.0
Mean Absolute Error:  44.2

Semi-global BM filled saved, using time:  23.9157049656
Bad matched pixel:    75.9
Mean Absolute Error:  52.4

Semi-global BM filled v1 saved, using time:  0.63592505455
Bad matched pixel:    76.1
Mean Absolute Error:  51.8

---------------------------------------------------------------
Building disparity map for stereo pair images2/Mask-perfect...

Image info:  (502, 698) 160
Semi-global BM saved, using time:  0.855034828186
Bad matched pixel:    72.4
Mean Absolute Error:  34.4

Image info:  (502, 698) 160
GC saved, using time:  169.356185913
Bad matched pixel:    72.4
Mean Absolute Error:  34.4

Image info:  (251, 349) 80
GC fast saved, using time:  15.8688058853
Bad matched pixel:    70.8
Mean Absolute Error:  36.3

Semi-global BM filled saved, using time:  24.4390451908
Bad matched pixel:    70.3
Mean Absolute Error:  39.8

Semi-global BM filled v1 saved, using time:  0.711218118668
Bad matched pixel:    70.1
Mean Absolute Error:  39.0

---------------------------------------------------------------
Building disparity map for stereo pair images2/Motorcycle-perfect...

Image info:  (500, 741) 112
Semi-global BM saved, using time:  0.917634963989
Bad matched pixel:    17.7
Mean Absolute Error:  4.9

Image info:  (500, 741) 112
GC saved, using time:  135.510635853
Bad matched pixel:    17.7
Mean Absolute Error:  4.9

Image info:  (250, 371) 48
GC fast saved, using time:  10.7279589176
Bad matched pixel:    15.3
Mean Absolute Error:  3.4

Semi-global BM filled saved, using time:  26.7944118977
Bad matched pixel:    13.2
Mean Absolute Error:  3.4

Semi-global BM filled v1 saved, using time:  0.588624000549
Bad matched pixel:    13.7
Mean Absolute Error:  3.5

---------------------------------------------------------------
Building disparity map for stereo pair images2/Piano-perfect...

Image info:  (480, 705) 96
Semi-global BM saved, using time:  0.523866891861
Bad matched pixel:    24.1
Mean Absolute Error:  5.7

Image info:  (480, 705) 96
GC saved, using time:  82.6582860947
Bad matched pixel:    24.1
Mean Absolute Error:  5.7

Image info:  (240, 353) 48
GC fast saved, using time:  8.05139493942
Bad matched pixel:    21.9
Mean Absolute Error:  4.2

Semi-global BM filled saved, using time:  31.3244969845
Bad matched pixel:    19.6
Mean Absolute Error:  3.8

Semi-global BM filled v1 saved, using time:  0.618901014328
Bad matched pixel:    19.0
Mean Absolute Error:  3.7

---------------------------------------------------------------
Building disparity map for stereo pair images2/Pipes-perfect...

Image info:  (481, 740) 128
Semi-global BM saved, using time:  0.895950078964
Bad matched pixel:    29.6
Mean Absolute Error:  8.2

Image info:  (481, 740) 128
GC saved, using time:  104.022825003
Bad matched pixel:    29.6
Mean Absolute Error:  8.2

Image info:  (241, 370) 64
GC fast saved, using time:  14.4630868435
Bad matched pixel:    33.5
Mean Absolute Error:  8.0

Semi-global BM filled saved, using time:  26.0162730217
Bad matched pixel:    24.4
Mean Absolute Error:  6.2

Semi-global BM filled v1 saved, using time:  0.588480949402
Bad matched pixel:    25.6
Mean Absolute Error:  6.3

---------------------------------------------------------------
Building disparity map for stereo pair images2/Playtable-perfect...

Image info:  (462, 681) 112
Semi-global BM saved, using time:  0.541335105896
Bad matched pixel:    21.7
Mean Absolute Error:  6.3

Image info:  (462, 681) 112
GC saved, using time:  202.287938118
Bad matched pixel:    21.7
Mean Absolute Error:  6.3

Image info:  (231, 341) 32
GC fast saved, using time:  13.6186320782
Bad matched pixel:    24.7
Mean Absolute Error:  4.2

Semi-global BM filled saved, using time:  22.3941998482
Bad matched pixel:    15.5
Mean Absolute Error:  3.6

Semi-global BM filled v1 saved, using time:  0.597630023956
Bad matched pixel:    16.1
Mean Absolute Error:  3.5

---------------------------------------------------------------

Process finished with exit code 0
'''

