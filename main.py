import cv2, os, time
import numpy as np
from skimage import segmentation, filters
from multiprocessing.pool import ThreadPool
from disparityMap import DisparityMap



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


def gen_disparity(imageL, imageR, ndisparities, postprocess=False):
    pool = ThreadPool(2)
    disp_map = pool.apply_async(build_disparity, args=(imageL, imageR, ndisparities))
    comp_map = pool.apply_async(build_segmentation, args=(imageL, ))
    pool.close()
    pool.join()
    result = fill_occlusion(disparity=disp_map.get(), segmentation=comp_map.get())

    if postprocess:
        result = cv2.medianBlur(result.astype(np.uint8), ksize=3)
    return result



def run_image(rootdir, save=False):
    for subdir, dirs, files in os.walk(rootdir):
        ok, imageL, imageR = False, None, None

        for f in files:
            filepath = os.path.join(subdir, f)

            if filepath.endswith('view1.png'):
                imageL = cv2.imread(filepath, 0)
                ok = True

            if filepath.endswith('view5.png'):
                imageR = cv2.imread(filepath, 0)

        if ok:
            t1 = time.time()
            for _ in range(10):
                result = gen_disparity(imageL=imageL, imageR=imageR, ndisparities=144, postprocess=True)
            t2 = time.time()
            if save:
                cv2.imwrite(os.path.join(subdir, 'output.png'), result)

            print 'Built disparity map for: %s\n image size: (%d, %d)\n running time: %.2f sec\n' \
                  %(subdir, imageL.shape[0], imageL.shape[1], (t2 - t1) / 10)




def run_video(filepath, display=True, save=False):
    if save:
        out = cv2.VideoWriter(filename=filepath + '_disparity.mp4', fourcc=cv2.cv.CV_FOURCC('M','J','P','G'), fps=5, frameSize=(960, 260), isColor=0)

    cap = cv2.VideoCapture(filepath)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        frame_idx += 1

        if ret == True:
            width = frame.shape[1] / 2
            imageL, imageR = frame[:, : width, :], frame[:, width:, :]

            imageL_gray = cv2.cvtColor(imageL, cv2.COLOR_BGR2GRAY)[50: -50]
            imageR_gray = cv2.cvtColor(imageR, cv2.COLOR_BGR2GRAY)[50: -50]

            if frame_idx >= 1200 and frame_idx <= 1900:
                t = time.time()
                result = gen_disparity(imageL=imageL_gray, imageR=imageR_gray, ndisparities=width/4, postprocess=True)
                print round(time.time() - t, 2)

                if display:
                    result_norm = cv2.normalize(result, alpha=0, beta=255, norm_type=cv2.cv.CV_MINMAX, dtype=cv2.cv.CV_8U)
                    output = cv2.hconcat((imageL_gray, result, result_norm))
                    cv2.imshow('Left View Disparity Map', output)

                if save:
                    out.write(output)

                # Press Q on keyboard to  exit
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':

    rootdir = 'stereo pair images/'
    # run_image(rootdir, save=False)

    run_video(filepath='videoplayback.mp4', display=True, save=False)

