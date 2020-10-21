import cv2

from numpy import hstack

from lib.common import KEYS
from lib.common import add_note_on_the_picture
from lib.filter import Filter
from lib.filter import methods


def show_task_1():
    """
    Load the image Lenna.png using OpenCV and display it side by side as both a
    grayscale image and a color image.
    :return: None
    """
    Lenna   = cv2.imread('../Data/Lenna.png')
    LennaBW = cv2.cvtColor(Lenna,   cv2.COLOR_BGR2GRAY)
    LennaBW = cv2.cvtColor(LennaBW, cv2.COLOR_GRAY2BGR)

    doubled_lenna = hstack((LennaBW, Lenna))
    add_note_on_the_picture(doubled_lenna)

    while cv2.waitKey(100) != KEYS.SPACE: cv2.imshow('Exercise 1.1: Lenna', doubled_lenna)

    cv2.destroyAllWindows()


def show_task_2():
    """
    Implement a change of functionality on key press:
        a) Change of color spaces (HSV, LAB, YUV)
        b) Adaptives thresholding in the variants Gaussian and Otsu-Thresholding.
        c) Canny edge extraction

    :return: None
    """

    cap = cv2.VideoCapture(0)
    transform = Filter.nothing
    while (pressed_key := cv2.waitKey(10)) != KEYS.Q:
        _, frame = cap.read()
        transform = methods[pressed_key] if pressed_key in methods.keys() else transform
        cv2.imshow('Exercise 1.2: OpenCV experiments', transform(frame))


def show_task_3():
    """
    implement a video streaming example showing SIFT features and visualize its keypoints as illustrated

    :return: None
    """

    cap = cv2.VideoCapture(0)
    while cv2.waitKey(10) != KEYS.Q:
        _, frame = cap.read()
        cv2.imshow('Exercise 1.3: SIFT features', Filter.sift_detection(frame))


if __name__ == "__main__":
    show_task_2()
