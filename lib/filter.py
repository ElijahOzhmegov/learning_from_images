import cv2
from datetime import datetime, timedelta

import numpy as np
from numpy import ndarray
from numpy.fft import fft2, ifft2

from enum import Enum
from lib.common import KEYS
from lib.common import add_note_on_the_picture


class State(Enum):
    ORIGINAL   = 0
    NORMALIZED = 1


class Helper:
    @staticmethod
    def make_gaussian(size, fwhm=3, center=None):
        """
        Make a square gaussian kernel.

        size is the length of a side of the square
        fwhm is full-width-half-maximum, which
        can be thought of as an effective radius.
        """

        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]

        if center is None:
            x0 = y0 = size // 2
        else:
            x0, y0 = center

        k = np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)
        return k / np.sum(k)


class Filter:

    @staticmethod
    def _normalize(img: ndarray):
        """
        Converts uint image (0-255) to double image (0.0-1.0) and generalizes
        this concept to any range.

        :param img:
        :return: normalized image
        """
        min_val = np.min(img.ravel())
        max_val = np.max(img.ravel())
        return (img.astype('uint8') - min_val) / (max_val - min_val)

    @staticmethod
    def _revert_normalization(img: ndarray):
        img = Filter._normalize(img) * 255
        return img.astype('int')

    @staticmethod
    def convolve(img: ndarray, kernel: ndarray):
        """
        Computes the convolution between kernel and image

        :param img: grayscale image
        :param kernel: convolution matrix - 3x3, or 5x5 matrix
        :return: result of the convolution

        source: https://laurentperrinet.github.io/sciblog/posts/2017-09-20-the-fastest-2d-convolution-in-the-world.html
        """
        A = img
        B = kernel

        return np.real(ifft2(fft2(A) * fft2(B, s=A.shape)))


    @staticmethod
    def nothing(img: ndarray):
        add_note_on_the_picture(img, "Nothing (key: 0)", label_center=(0, 0))
        return img

    @staticmethod
    def gaussian_blur(img: ndarray, kernel_size=5):
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        add_note_on_the_picture(img, "Gaussian Blur (key: 9)", label_center=(0, 0))
        return img

    @staticmethod
    def to_hsv(img: ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        add_note_on_the_picture(img, "HSV colour space (key: 1)", label_center=(0, 0))
        return img

    @staticmethod
    def to_lab(img: ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        add_note_on_the_picture(img, "LAB colour space (key: 2)", label_center=(0, 0))
        return img

    @staticmethod
    def to_yuv(img: ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        add_note_on_the_picture(img, "YUV colour space (key: 3)", label_center=(0, 0))
        return img

    @staticmethod
    def adaptive_gaussian_thresholding(img: ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(img, 5)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        add_note_on_the_picture(img, "Adaptive Gaussian thresholding (key: 4)", label_center=(0, 0))
        return img

    @staticmethod
    def otsu_thresholding(img: ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        add_note_on_the_picture(otsu, "Otsu's thresholding (key: 5)", label_center=(0, 0))
        return otsu

    @staticmethod
    def canny_edge_detection(img: ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.Canny(img, 100, 200)
        add_note_on_the_picture(img, "Canny Edge Detection (key: 6)", label_center=(0, 0))
        return img

    @staticmethod
    def sift_detection(img: ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(gray)
        cv2.drawKeypoints(gray, kp, img)
        add_note_on_the_picture(img, "SIFT Detection (key: 7)", label_center=(0, 0))
        return img

    @staticmethod
    def my_sobel_edge_detection(img: ndarray):
        before = datetime.now()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = Filter._normalize(img)

        gauss_filter = Helper.make_gaussian(5)
        sobelmask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobelmask_y = np.array([[1, 2, 1],  [0, 0, 0],  [-1, -2, -1]])

        img = Filter.convolve(img, gauss_filter)
        sobel_x = Filter.convolve(img, sobelmask_x)
        sobel_y = Filter.convolve(img, sobelmask_y)
        sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        sobel = np.ascontiguousarray(sobel)

        after = datetime.now()
        diff = 1e6/(after - before).microseconds
        add_note_on_the_picture(sobel, "FPS " + str(round(diff, 2)), label_center=(0, 0))

        return sobel

    @staticmethod
    def sobel_edge_detection(img: ndarray):
        before = datetime.now()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = Filter._normalize(img)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        sobelmask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobelmask_y = np.array([[1, 2, 1],  [0, 0, 0],  [-1, -2, -1]])

        sobel_x = cv2.filter2D(img, -1, sobelmask_x)
        sobel_y = cv2.filter2D(img, -1, sobelmask_y)

        sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        sobel = np.ascontiguousarray(sobel)

        after = datetime.now()
        diff = 1e6/(after - before).microseconds
        add_note_on_the_picture(sobel, "FPS " + str(round(diff, 2)), label_center=(0, 0))

        return sobel


methods = {KEYS.ZERO:  Filter.nothing,
           KEYS.ONE:   Filter.to_hsv,
           KEYS.TWO:   Filter.to_lab,
           KEYS.THREE: Filter.to_yuv,
           KEYS.FOUR:  Filter.adaptive_gaussian_thresholding,
           KEYS.FIVE:  Filter.otsu_thresholding,
           KEYS.SIX:   Filter.canny_edge_detection,
           KEYS.SEVEN: Filter.sift_detection,
           KEYS.NINE:  Filter.gaussian_blur}
