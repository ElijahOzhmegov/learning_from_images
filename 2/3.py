import cv2
import numpy as np

from lib.common import KEYS
from lib.common import load_images
from lib.common import add_note_on_the_picture


def myCornerHaris(gray, k: float = 0.04, threshold: float= 0.01):
    smask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1,  0,  1]])
    smask_y = np.array([[1,  2, 1], [0,  0, 0], [-1, -2, -1]])

    G_x = cv2.filter2D(gray, -1, smask_x)
    G_y = cv2.filter2D(gray, -1, smask_y)

    G_xx = G_x ** 2
    G_xy = G_x * G_y
    G_yy = G_y ** 2

    G_xx, G_xy, G_yy = [cv2.filter2D(G, -1, np.ones((3, 3))) for G in [G_xx, G_xy, G_yy]]

    detM = G_xx*G_yy - G_xy**2
    TrM = G_xx + G_yy
    harris = detM - k * TrM ** 2

    result = np.zeros(harris.shape)
    result[harris > threshold*harris.max()] = [255]

    return result


if __name__ == "__main__":

    for img_path in load_images('../Data/images/harris/*'):
        img, path = img_path

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = np.float32(gray)

        k = 0.04
        threshold = 0.01

        harris_cv = cv2.cornerHarris(gray, 3, 3, k)
        my_harris = myCornerHaris(gray, k, threshold)

        harris_cv_thres = np.zeros(harris_cv.shape)
        harris_cv_thres[harris_cv > threshold * harris_cv.max()] = [255]

        img[my_harris == 255] = [0, 255, 0]
        diff = np.sum(np.absolute(my_harris - harris_cv_thres))

        add_note_on_the_picture(harris_cv_thres, text="OpenCV Corners")
        add_note_on_the_picture(my_harris,       text="My Corners")
        add_note_on_the_picture(img, text="diff: " + str(diff))

        res = cv2.hconcat((harris_cv_thres.astype(np.uint8), my_harris.astype(np.uint8)))
        res = cv2.hconcat([cv2.cvtColor(res, cv2.COLOR_GRAY2RGB), img])

        while cv2.waitKey(10) != KEYS.SPACE:
            cv2.imshow('Hariss Corner Detection (SPACE TO SHOW NEXT)', res)









