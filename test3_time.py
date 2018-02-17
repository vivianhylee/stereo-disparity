import cv2
import os
import time
from multiprocessing.pool import ThreadPool
import numpy as np
from disparityMap import DisparityMap
from evaluationDM import EvaluationDM
from segmentation import run_segmentation
from skimage import segmentation, color, filters



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

            # Semi-global block matching
            # t = 0
            # for _ in range(10):
            #     temp = time.time()
            #
            #     dm = DisparityMap(imageLeft=imageL, imageRight=imageR, ndisparities=144)
            #     dmBM = dm.semiGlobalBlockMatch()
            #
            #     t += time.time() - temp
            #
            # print 'Semi-global BM using time: ', t / 10



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
            t = 0
            for _ in range(10):
                temp = time.time()

                dm = DisparityMap(imageLeft=imageL, imageRight=imageR, ndisparities=128)
                disparity = dm.semiGlobalBlockMatch()
                comp_map = segmentation.watershed(filters.sobel(imageL), markers=100, compactness=0.001)
                disparity_filled_v3 = fill_occlusion(disparity=disparity, segmentation=comp_map)

                t += time.time() - temp

            print 'Semi-global BM filled v3 using time: ', t / 10


            t = 0
            imageLs = scaleDown(imageL, 2)
            imageRs = scaleDown(imageR, 2)
            for _ in range(10):
                temp = time.time()

                dm = DisparityMap(imageLeft=imageLs, imageRight=imageRs, ndisparities=48)
                disparity = dm.semiGlobalBlockMatch()
                comp_map = segmentation.watershed(filters.sobel(imageLs), markers=100, compactness=0.001)
                disparity_filled_v3 = fill_occlusion(disparity=disparity, segmentation=comp_map)

                t += time.time() - temp

            print 'Semi-global BM filled v3 using time: ', t / 10


            t = 0
            for _ in range(10):
                temp = time.time()

                pool = ThreadPool(2)
                r1 = pool.apply_async(building_map, args=(imageL, imageR, 128))
                r2 = pool.apply_async(segment, args=(imageL, ))
                pool.close()
                pool.join()
                disparity_filled_v3 = fill_occlusion(disparity=r1.get(), segmentation=r2.get())

                t += time.time() - temp

            print 'Semi-global BM filled v3 using time: ', t / 10


            t = 0
            imageLs = scaleDown(imageL, 2)
            imageRs = scaleDown(imageR, 2)
            for _ in range(10):
                temp = time.time()

                pool = ThreadPool(2)
                r1 = pool.apply_async(building_map, args=(imageLs, imageRs, 48))
                r2 = pool.apply_async(segment, args=(imageLs,))
                pool.close()
                pool.join()
                disparity_filled_v3 = fill_occlusion(disparity=r1.get(), segmentation=r2.get())

                t += time.time() - temp

            print 'Semi-global BM filled v3 using time: ', t / 10


        print '---------------------------------------------------------------'



def building_map(imageL, imageR, ndisparities):
    dm = DisparityMap(imageLeft=imageL, imageRight=imageR, ndisparities=ndisparities)
    disparity = dm.semiGlobalBlockMatch()
    return disparity

def segment(imageL):
    comp =segmentation.watershed(filters.sobel(imageL), markers=100, compactness=0.001)
    return comp



if __name__ == '__main__':
    run()





'''
original code

/usr/local/Cellar/python/2.7.13_1/Frameworks/Python.framework/Versions/2.7/bin/python2.7 /Users/hsiang-yunlee/LucidVR/test3_time.py
---------------------------------------------------------------
Building disparity map for stereo pair images/Aloe...

Semi-global BM using time:  1.5719738245
Semi-global BM filled v3 using time:  2.14664523602
---------------------------------------------------------------
Building disparity map for stereo pair images/Art...

Semi-global BM using time:  0.858477878571
Semi-global BM filled v3 using time:  1.39668321609
---------------------------------------------------------------
Building disparity map for stereo pair images/Baby1...

Semi-global BM using time:  0.671728920937
Semi-global BM filled v3 using time:  1.0654104948
---------------------------------------------------------------
Building disparity map for stereo pair images/Baby2...

Semi-global BM using time:  0.627123427391
Semi-global BM filled v3 using time:  1.05673632622
---------------------------------------------------------------
Building disparity map for stereo pair images/Baby3...

Semi-global BM using time:  0.776899909973
Semi-global BM filled v3 using time:  1.10985560417
---------------------------------------------------------------
Building disparity map for stereo pair images/Books...

Semi-global BM using time:  0.910110092163
Semi-global BM filled v3 using time:  1.32631206512
---------------------------------------------------------------
Building disparity map for stereo pair images/Bowling1...

Semi-global BM using time:  0.597861981392
Semi-global BM filled v3 using time:  1.010663414
---------------------------------------------------------------
Building disparity map for stereo pair images/Bowling2...

Semi-global BM using time:  0.70109360218
Semi-global BM filled v3 using time:  1.15083734989
---------------------------------------------------------------
Building disparity map for stereo pair images/Cloth1...

Semi-global BM using time:  2.25899379253
Semi-global BM filled v3 using time:  2.73316607475
---------------------------------------------------------------
Building disparity map for stereo pair images/Cloth2...

Semi-global BM using time:  0.915450763702
Semi-global BM filled v3 using time:  1.36402192116
---------------------------------------------------------------
Building disparity map for stereo pair images/Cloth3...

Semi-global BM using time:  0.993999361992
Semi-global BM filled v3 using time:  1.38988554478
---------------------------------------------------------------
Building disparity map for stereo pair images/Cloth4...

Semi-global BM using time:  1.48631350994
Semi-global BM filled v3 using time:  1.96895668507
---------------------------------------------------------------
Building disparity map for stereo pair images/Dolls...

Semi-global BM using time:  0.94756872654
Semi-global BM filled v3 using time:  1.40692756176
---------------------------------------------------------------
Building disparity map for stereo pair images/Flowerpots...

Semi-global BM using time:  0.605770921707
Semi-global BM filled v3 using time:  1.03667731285
---------------------------------------------------------------
Building disparity map for stereo pair images/Lampshade1...

Semi-global BM using time:  0.554349851608
Semi-global BM filled v3 using time:  0.947836089134
---------------------------------------------------------------
Building disparity map for stereo pair images/Lampshade2...

Semi-global BM using time:  0.463611578941
Semi-global BM filled v3 using time:  0.901937031746
---------------------------------------------------------------
Building disparity map for stereo pair images/Laundry...

Semi-global BM using time:  0.768880414963
Semi-global BM filled v3 using time:  1.20934844017
---------------------------------------------------------------
Building disparity map for stereo pair images/Midd1...

Semi-global BM using time:  0.730255079269
Semi-global BM filled v3 using time:  1.24036672115
---------------------------------------------------------------
Building disparity map for stereo pair images/Midd2...

Semi-global BM using time:  0.698141884804
Semi-global BM filled v3 using time:  1.16467170715
---------------------------------------------------------------
Building disparity map for stereo pair images/Moebius...

Semi-global BM using time:  0.78588757515
Semi-global BM filled v3 using time:  1.2380751133
---------------------------------------------------------------
Building disparity map for stereo pair images/Monopoly...

Semi-global BM using time:  0.696973633766
Semi-global BM filled v3 using time:  1.1331171751
---------------------------------------------------------------
Building disparity map for stereo pair images/Plastic...

Semi-global BM using time:  0.535763144493
Semi-global BM filled v3 using time:  1.02625293732
---------------------------------------------------------------
Building disparity map for stereo pair images/Reindeer...

Semi-global BM using time:  0.721308779716
Semi-global BM filled v3 using time:  1.17260565758
---------------------------------------------------------------
Building disparity map for stereo pair images/Rocks1...

Semi-global BM using time:  0.858949780464
Semi-global BM filled v3 using time:  1.28444440365
---------------------------------------------------------------
Building disparity map for stereo pair images/Rocks2...

Semi-global BM using time:  0.858272099495
Semi-global BM filled v3 using time:  1.29174830914
---------------------------------------------------------------
Building disparity map for stereo pair images/Wood1...

Semi-global BM using time:  0.636512255669
Semi-global BM filled v3 using time:  1.09201099873
---------------------------------------------------------------
Building disparity map for stereo pair images/Wood2...

Semi-global BM using time:  0.615046143532
Semi-global BM filled v3 using time:  1.05621211529
---------------------------------------------------------------

Process finished with exit code 0
'''

'''
using default number of disparities


/usr/local/Cellar/python/2.7.13_1/Frameworks/Python.framework/Versions/2.7/bin/python2.7 /Users/hsiang-yunlee/LucidVR/test3_time.py
---------------------------------------------------------------
Building disparity map for stereo pair images/Aloe...

Semi-global BM using time:  0.390671491623
Semi-global BM filled v3 using time:  0.928074455261
---------------------------------------------------------------
Building disparity map for stereo pair images/Art...

Semi-global BM using time:  0.398108530045
Semi-global BM filled v3 using time:  0.933099389076
---------------------------------------------------------------
Building disparity map for stereo pair images/Baby1...

Semi-global BM using time:  0.361698412895
Semi-global BM filled v3 using time:  0.751889133453
---------------------------------------------------------------
Building disparity map for stereo pair images/Baby2...

Semi-global BM using time:  0.371870803833
Semi-global BM filled v3 using time:  0.773922801018
---------------------------------------------------------------
Building disparity map for stereo pair images/Baby3...

Semi-global BM using time:  0.401076793671
Semi-global BM filled v3 using time:  0.816241168976
---------------------------------------------------------------
Building disparity map for stereo pair images/Books...

Semi-global BM using time:  0.478053092957
Semi-global BM filled v3 using time:  0.961491394043
---------------------------------------------------------------
Building disparity map for stereo pair images/Bowling1...

Semi-global BM using time:  0.405355381966
Semi-global BM filled v3 using time:  0.806684398651
---------------------------------------------------------------
Building disparity map for stereo pair images/Bowling2...

Semi-global BM using time:  0.396448302269
Semi-global BM filled v3 using time:  0.841235113144
---------------------------------------------------------------
Building disparity map for stereo pair images/Cloth1...

Semi-global BM using time:  0.351209783554
Semi-global BM filled v3 using time:  0.763862085342
---------------------------------------------------------------
Building disparity map for stereo pair images/Cloth2...

Semi-global BM using time:  0.363010191917
Semi-global BM filled v3 using time:  0.797405433655
---------------------------------------------------------------
Building disparity map for stereo pair images/Cloth3...

Semi-global BM using time:  0.351356816292
Semi-global BM filled v3 using time:  0.745668745041
---------------------------------------------------------------
Building disparity map for stereo pair images/Cloth4...

Semi-global BM using time:  0.361932277679
Semi-global BM filled v3 using time:  0.826352572441
---------------------------------------------------------------
Building disparity map for stereo pair images/Dolls...

Semi-global BM using time:  0.395370721817
Semi-global BM filled v3 using time:  0.847792339325
---------------------------------------------------------------
Building disparity map for stereo pair images/Flowerpots...

Semi-global BM using time:  0.365408420563
Semi-global BM filled v3 using time:  0.826816082001
---------------------------------------------------------------
Building disparity map for stereo pair images/Lampshade1...

Semi-global BM using time:  0.409317708015
Semi-global BM filled v3 using time:  0.789488172531
---------------------------------------------------------------
Building disparity map for stereo pair images/Lampshade2...

Semi-global BM using time:  0.375764942169
Semi-global BM filled v3 using time:  0.79251947403
---------------------------------------------------------------
Building disparity map for stereo pair images/Laundry...

Semi-global BM using time:  0.391812753677
Semi-global BM filled v3 using time:  0.811123538017
---------------------------------------------------------------
Building disparity map for stereo pair images/Midd1...

Semi-global BM using time:  0.40972571373
Semi-global BM filled v3 using time:  0.89753446579
---------------------------------------------------------------
Building disparity map for stereo pair images/Midd2...

Semi-global BM using time:  0.395775437355
Semi-global BM filled v3 using time:  0.854638409615
---------------------------------------------------------------
Building disparity map for stereo pair images/Moebius...

Semi-global BM using time:  0.40350985527
Semi-global BM filled v3 using time:  0.856105136871
---------------------------------------------------------------
Building disparity map for stereo pair images/Monopoly...

Semi-global BM using time:  0.389505815506
Semi-global BM filled v3 using time:  0.780111765862
---------------------------------------------------------------
Building disparity map for stereo pair images/Plastic...

Semi-global BM using time:  0.367806482315
Semi-global BM filled v3 using time:  0.860607457161
---------------------------------------------------------------
Building disparity map for stereo pair images/Reindeer...

Semi-global BM using time:  0.397590136528
Semi-global BM filled v3 using time:  0.865688800812
---------------------------------------------------------------
Building disparity map for stereo pair images/Rocks1...

Semi-global BM using time:  0.381614565849
Semi-global BM filled v3 using time:  0.855469226837
---------------------------------------------------------------
Building disparity map for stereo pair images/Rocks2...

Semi-global BM using time:  0.372347688675
Semi-global BM filled v3 using time:  0.821570849419
---------------------------------------------------------------
Building disparity map for stereo pair images/Wood1...

Semi-global BM using time:  0.416065144539
Semi-global BM filled v3 using time:  0.869602608681
---------------------------------------------------------------
Building disparity map for stereo pair images/Wood2...

Semi-global BM using time:  0.415529966354
Semi-global BM filled v3 using time:  0.872425365448
---------------------------------------------------------------

Process finished with exit code 0
'''