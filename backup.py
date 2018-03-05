import cv2, os, time
import numpy as np
from multiprocessing.pool import ThreadPool
from disparityMap import DisparityMap
from evaluationDM import EvaluationDM
from skimage import segmentation, filters



def scale_image(image, ratio=2.):
    # other interpolation: cv2.INTER_NEAREST
    return cv2.resize(image, None, fx=1./ratio, fy=1./ratio, interpolation=cv2.INTER_LANCZOS4)


def build_disparity(imageL, imageR, ndisparities):
    dm = DisparityMap(imageLeft=imageL, imageRight=imageR, ndisparities=ndisparities)
    disparity = dm.semiGlobalBlockMatch()
    return disparity


def build_segmentation(imageL):
    component_map = segmentation.watershed(filters.sobel(imageL), markers=100, compactness=0.00001)
    return component_map


def fill_occlusion(disparity, segmentation):
    output = np.copy(disparity)
    comp_index = {}
    # coordinates of occlusion area
    rows, cols, = np.where(disparity == 0)
    for r, c in zip(rows, cols):
        id = segmentation[r, c]

        if id not in comp_index:
            # using majority pixel intensity from the area in the same segmentation area to fill occlusion pixel
            coords = np.where(segmentation == id)
            vals = disparity[coords]
            (values, counts) = np.unique(vals[np.where(vals > 0)], return_counts=True)
            comp_index[id] = 0 if len(counts) == 0 else values[np.argmax(counts)]

        output[r, c] = comp_index[id]
    return output


def gen_disparity(imageL, imageR, ndisparities, post_processing=False):
    pool = ThreadPool(2)
    disp_map = pool.apply_async(build_disparity, args=(imageL, imageR, ndisparities))
    comp_map = pool.apply_async(build_segmentation, args=(imageL, ))
    pool.close()
    pool.join()
    result = fill_occlusion(disparity=disp_map.get(), segmentation=comp_map.get())

    if post_processing:
        result = cv2.medianBlur(result.astype(np.uint8), ksize=3)
    return result


def evaluate(result, imageGT):
    error = EvaluationDM(imageGT=imageGT, imageET=[result])
    error.evaluate()
    return error


def run_eval(imageL, imageR, imageGT, ndisparities, isDavide=False, cycle=10, print_out=False):
    total, result = 0, None
    for _ in range(cycle):
        temp = time.time()
        result = run(imageL=imageL, imageR=imageR, ndisparities=ndisparities, isDivide=isDavide)
        total += time.time() - temp
    if print_out:
        print 'Image dimension: ', imageL.shape, ' using running time: ', round(total / cycle, 2)
        print evaluate(result=result, imageGT=imageGT)
    return result


def run(imageL, imageR, ndisparities, isDivide=False):
    if isDivide:
        height, width = imageL.shape[: 2]
        num_strips = height / 100
        pool = ThreadPool(num_strips)
        threads = []
        for i in range(num_strips):
            if i == num_strips - 1:
                r = pool.apply_async(gen_disparity, args=(imageL[i * height / num_strips:, :], imageR[i * height / num_strips:, :], ndisparities))
            else:
                r = pool.apply_async(gen_disparity, args=(imageL[i * height / num_strips: (i + 1) * height / num_strips, :],
                                                          imageR[i * height / num_strips: (i + 1) * height / num_strips, :],
                                                          ndisparities))
            threads.append(r)

        pool.close()
        pool.join()
        result = np.concatenate([threads[i].get() for i in range(num_strips)], axis=0)

    else:
        result = gen_disparity(imageL=imageL, imageR=imageR, ndisparities=ndisparities)

    # result_blur = post_processing(result)
    return result



def run_video(ipt_filepath, out_filepath):
    out = cv2.VideoWriter(filename=out_filepath, fourcc=cv2.cv.CV_FOURCC('M','J','P','G'), fps=5, frameSize=(960, 260), isColor=0)
    cap = cv2.VideoCapture(ipt_filepath)
    frame_idx = 0

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_idx += 1

        if ret == True:
            width = frame.shape[1] / 2
            imageL, imageR = frame[:, : width, :], frame[:, width:, :]

            imageL_gray = cv2.cvtColor(imageL, cv2.COLOR_BGR2GRAY)[50: -50]
            imageR_gray = cv2.cvtColor(imageR, cv2.COLOR_BGR2GRAY)[50: -50]

            if frame_idx >= 1200 and frame_idx <= 1900:
                result = run(imageL=imageL_gray, imageR=imageR_gray, ndisparities=width / 4)

                # Display the resulting frame
                result_norm = cv2.normalize(result, alpha=0, beta=255, norm_type=cv2.cv.CV_MINMAX, dtype=cv2.cv.CV_8U)
                comb = cv2.hconcat((imageL_gray, result, result_norm))
                cv2.imshow('Left View Disparity Map', comb)

                # Save the resulting frame
                out.write(comb)

                # Press Q on keyboard to  exit
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

        # Break the loop
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':

    rootdir = 'stereo pair images/'

    for subdir, dirs, files in os.walk(rootdir):
        ok, imageGT, imageL, imageR = False, None, None, None

        for f in files:
            filepath = os.path.join(subdir, f)

            if filepath.endswith('disp1.png'):
                imageGT = cv2.imread(filepath, 0) * 0.5
                ok = True

            elif filepath.endswith('view1.png'):
                imageL = cv2.imread(filepath, 0)

            elif filepath.endswith('view5.png'):
                imageR = cv2.imread(filepath, 0)


    # # rootdir = 'data_scene_flow_12/training/'
    # # rootdirGT = rootdir + 'disp_occ'
    # # rootdirLeft = rootdir + 'image_0'
    # # rootdirRight = rootdir + 'image_1'
    # #
    # # for file in os.listdir(rootdirLeft):
    # #     ok, subdir, imageGT, imageLeft, imageRight = False, None, None, None, None
    # #     if file.endswith('_10.png'):
    # #         imageGT = cv2.imread(os.path.join(rootdirGT, file), 0)
    # #         imageL = cv2.imread(os.path.join(rootdirLeft, file), 0)
    # #         imageR = cv2.imread(os.path.join(rootdirRight, file), 0)
    # #         ok = True
    # #         subdir = rootdir + 'output/'
    #
    #
    #
        # if ok:
            # print 'Building disparity map for ' + subdir + '...\n'

            # Improved Semi-global block matching using segmentation watershed
            # result = run_eval(imageL=imageL, imageR=imageR, ndisparities=160, imageGT=imageGT, cycle=10, print_out=True)
            # print evaluate(result=result, imageGT=imageGT)
            #
            # output = np.vstack((result, result))
            # output[: result.shape[0], : result.shape[1]] = 0

            # # Improved Semi-global block matching using segmentation watershed with downsize
            # ratio = 2.0
            # imageLs = scaleDown(image=imageL, ratio=ratio)
            # imageRs = scaleDown(image=imageR, ratio=ratio)
            # imageGTs = scaleDown(image=imageGT, ratio=ratio)
            # result = run_eval(imageL=imageLs, imageR=imageRs, ndisparities=80, imageGT=imageGTs, cycle=10, print_out=True) * ratio
            # print evaluate(result=result, imageGT=imageGTs)

            # output[: result.shape[0], : result.shape[1]] = result
            # cv2.imwrite(os.path.join(subdir, file[: -4] + '_comb1.png'), output)

        # print '---------------------------------------------------------------'


        if ok:
            print 'Building disparity map for ' + subdir + '...\n'
            result = run_eval(imageL=imageL, imageR=imageR, ndisparities=160, imageGT=imageGT, cycle=10, print_out=True, isDavide=False)

            result = run_eval(imageL=imageL, imageR=imageR, ndisparities=160, imageGT=imageGT, cycle=10, print_out=True, isDavide=True)
            # cv2.imwrite(os.path.join(subdir, 'disparity_BM_filled_v3_divide.png'), result)


        # run_video('videoplayback.mp4', 'out.mp4')

