import cv2

from numpy import hstack
from lib.common import SPACE_KEY
from lib.common import add_note_on_the_picture


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

    while cv2.waitKey(100) != SPACE_KEY: cv2.imshow('Exercise 1.1: Lenna', doubled_lenna)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_task_1()
