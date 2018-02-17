import cv2, os
import numpy as np
from evaluationDM import EvaluationDM


def plot(dataX, dataY, labelX="", labelY="", title=""):
    import matplotlib.pyplot as plt

    plt.plot(dataX, dataY, 'ro')
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.title(title)
    return plt



if __name__ == '__main__':
    # rootdir = 'stereo pair images/'
    #
    # filename = np.array(
    #     ['GT.png', 'BM.png', 'GC.png', 'GC_fast.png', 'BM_filled.png', 'BM_filled_v1.png', 'BM_filled_v2.png',
    #      'BM_filled_v3.png'])
    #
    # file = open(rootdir + 'performance.txt', 'w')
    #
    # for subdir, dirs, files in os.walk(rootdir):
    #     run, disparities = False, [None] * len(filename)
    #
    #     for f in files:
    #         filepath = os.path.join(subdir, f)
    #         for i, fname in enumerate(filename):
    #             if filepath.endswith(fname):
    #                 disparities[i] = cv2.imread(filepath, 0)
    #                 run = True
    #
    #     if run:
    #         comb1 = np.concatenate(disparities[: 4], axis=1)
    #         comb2 = np.concatenate(disparities[4: ], axis=1)
    #         cv2.imwrite(os.path.join(subdir, 'comb.png'), np.concatenate((comb1, comb2), axis=0))
    #
    #         # print subdir
    #         evaluator = EvaluationDM(imageGT=disparities[0], imageET=disparities[1:])
    #         evaluator.evaluate()
    #         result = evaluator.report
    #         # print 'Bad pixel %: ', result[0]
    #         # print 'MAE        : ', result[1]
    #         # print
    #
    #         file.write(subdir + "\n")
    #         file.write(np.array2string(filename[1:]) + "\n")
    #         file.write("BMP(%)\n" + np.array2string(result[0]) + "\n")
    #         file.write("MAE(%)\n" + np.array2string(result[1]) + "\n")
    #         file.write("\n")
    #
    # print "result saved to file at: " + rootdir
    # file.close()



    rootdir = 'data_scene_flow_12/training/output'

    filename = np.array(
        ['BM.png', 'GC.png', 'GC_fast.png', 'BM_filled.png', 'BM_filled_v1.png', 'BM_filled_v2.png',
         'BM_filled_v3.png'])

    # file = open(rootdir + 'performance.txt', 'w')
    for i in range(0, 43):
        fname = ('000000' + str(i))[-6: ] + '_10_'
        disparities = [cv2.imread(os.path.join(rootdir, fname + ext), 0) for ext in filename]

        if len(disparities) == len(filename):
            comb = np.concatenate(disparities, axis=0)
            cv2.imwrite(os.path.join(rootdir, fname + 'comb.png'), comb)
            print i

