import cv2
import os
import time
import numpy as np
from disparityMap import DisparityMap
from evaluationDM import EvaluationDM
from segmentation import run_segmentation


def scaleDown(image, ratio=2):
    height, width = image.shape[: 2]
    output = np.copy(image)
    return cv2.pyrDown(output, (height / ratio, width / ratio))


def scaleUp(image, ratio=2):
    m, n = image.shape[: 2]
    return cv2.pyrUp(image, (m * ratio, n * ratio))


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
    rows, cols, = np.where(disparity < 5)
    for r, c in zip(rows, cols):
        id = segmentation[r, c]

        if id not in index:
            coords = np.where(segmentation== id)
            vals = disparity[coords]
            (values, counts) = np.unique(vals[np.where(vals > 5)], return_counts=True)

            if len(counts) == 0:
                index[id] = 0
            else:
                index[id] = values[np.argmax(counts)]

        output[r, c] = index[id]

    return output


def run():
    rootdir = 'stereo pair images/'


    for subdir, dirs, files in os.walk(rootdir):

        run, imageGT, imageLeft, imageRight = False, None, None, None

        for f in files:
            filepath = os.path.join(subdir, f)

            if filepath.endswith('disp1.png'):
                imageGT = cv2.imread(filepath, 0) * 0.5
                run = True

            elif filepath.endswith('view1.png'):
                imageL = cv2.imread(filepath, 0)

            elif filepath.endswith('view5.png'):
                imageR = cv2.imread(filepath, 0)


    # rootdir = 'data_scene_flow/training/'
    # rootdirGT    = rootdir + 'disp_noc_0'
    # rootdirLeft  = rootdir + 'image_2'
    # rootdirRight = rootdir + 'image_3'
    #
    # for file in os.listdir(rootdirLeft):
    #     run, subdir, imageGT, imageLeft, imageRight = False, None, None, None, None
    #     if file.endswith('_10.png'):
    #         imageGT = cv2.imread(os.path.join(rootdirGT,    file), 0)
    #         imageL  = cv2.imread(os.path.join(rootdirLeft,  file), 0)
    #         imageR  = cv2.imread(os.path.join(rootdirRight, file), 0)
    #         run = True
    #         subdir = rootdir + 'output/'

        if run:
            print 'Building disparity map for ' + subdir + '...\n'

            # Semi-global block matching
            t1 = time.time()
            dm = DisparityMap(imageLeft=imageL, imageRight=imageR)
            dmBM = dm.semiGlobalBlockMatch()

            #filename = file[: -4] + '_BM.png'
            filename = 'disparity_BM.png'
            cv2.imwrite(os.path.join(subdir, filename), dmBM)
            print 'Semi-global BM saved, using time: ', time.time() - t1

            error = EvaluationDM(imageGT=imageGT, imageET=dmBM)
            print 'Bad matched pixel:   ', error.BMP()
            print 'Mean Absolute Error: ', error.MAE()
            print


            # Graph cut
            t1 = time.time()
            dm = DisparityMap(imageLeft=imageL, imageRight=imageR)
            dmGC = dm.graphCut()

            #filename = file[: -4] + '_GC.png'
            filename = 'disparity_GC.png'
            cv2.imwrite(os.path.join(subdir, filename), dmGC)
            print 'GC saved, using time: ', time.time() - t1

            error = EvaluationDM(imageGT=imageGT, imageET=dmGC)
            print 'Bad matched pixel:   ', error.BMP()
            print 'Mean Absolute Error: ', error.MAE()
            print


            # Graph cut fast by downsize image
            ratio = 2
            imageL_scale = scaleDown(image=imageL, ratio=ratio)
            imageR_scale = scaleDown(image=imageR, ratio=ratio)

            t1 = time.time()
            dm = DisparityMap(imageLeft=imageL_scale, imageRight=imageR_scale)
            dmGC_fast = dm.graphCut() * ratio
            dmGC_fast = scaleUp(dmGC_fast)
            dmGC_fast = matchSize(image1=imageGT, image2=dmGC_fast)

            #filename = file[: -4] + '_GC_fast.png'
            filename = 'disparity_GC_fast.png'
            cv2.imwrite(os.path.join(subdir, filename), dmGC_fast)
            print 'GC fast saved, using time: ', time.time() - t1

            error = EvaluationDM(imageGT=imageGT, imageET=dmGC_fast)
            print 'Bad matched pixel:   ', error.BMP()
            print 'Mean Absolute Error: ', error.MAE()
            print


            # Improve Semi-global block matching using segmentation
            t1 = time.time()
            comp_map = run_segmentation(image=imageL)
            comp_map = np.asarray(comp_map)

            disparity = dmBM

            disparity_filled = fill_occlusion(disparity=disparity, segmentation=comp_map)

            #filename = file[: -4] + '_BM_filled.png'
            filename = 'disparity_BM_filled.png'
            cv2.imwrite(os.path.join(subdir, filename), disparity_filled)
            print 'Semi-global BM filled saved, using time: ', time.time() - t1

            error = EvaluationDM(imageGT=imageGT, imageET=disparity_filled)
            print 'Bad matched pixel:   ', error.BMP()
            print 'Mean Absolute Error: ', error.MAE()
            print

        print '---------------------------------------------------------------'





if __name__ == '__main__':
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