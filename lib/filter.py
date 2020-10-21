import cv2
from numpy import ndarray
from lib.common import KEYS
from lib.common import add_note_on_the_picture


class Filter:

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


methods = {KEYS.ZERO:  Filter.nothing,
           KEYS.ONE:   Filter.to_hsv,
           KEYS.TWO:   Filter.to_lab,
           KEYS.THREE: Filter.to_yuv,
           KEYS.FOUR:  Filter.adaptive_gaussian_thresholding,
           KEYS.FIVE:  Filter.otsu_thresholding,
           KEYS.SIX:   Filter.canny_edge_detection,
           KEYS.SEVEN: Filter.sift_detection,
           KEYS.NINE:  Filter.gaussian_blur}
