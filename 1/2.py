import numpy as np
import cv2
import math
import sys

from numpy import ndarray
from enum import Enum
from lib.common import KEYS
from lib.common import add_note_on_the_picture
from lib.common import k_to_rgb
from lib.filter import Filter


class Render(Enum):
    CLUSTERS_CENTERS = 0
    BLACK_AND_WHITE = 1
    BRIGHT_COLOURS = 2


class Kmeans():

    def __init__(self, img, k=3, break_eps=.001):
        self.k = k
        self.img = img.copy()

        self.n_layers = img.shape[-1]
        self.cluster_centers = np.random.randint(255, size=(k, img.shape[-1]))
        self.clustered_img = np.random.randint(k, size=img.shape[:2])

        self.break_eps = break_eps
        self.error = 0
        self.eps = 1.

        print("Initialised")
        self.recluster()

    def calculate_cluster_centers(self):

        for i in range(self.k):
            k_elements = self.clustered_img == i
            for j in range(self.n_layers):
                L = self.img[k_elements, j]
                # try:
                self.cluster_centers[i, j] = np.mean(L)
                # except:
                #     continue

        print("Cluster centers recalculated")

    def recluster(self):
        if self.eps < self.break_eps: return

        n, m, _ = self.img.shape
        pixel_distances_to_means = self.cluster_centers[:, 0]
        prev_error = self.error
        self.error = 0

        for i in range(n):
            for j in range(m):

                for k in range(self.k):
                    pixel_distances_to_means[k] = self.distance(self.img[i][j], self.cluster_centers[k])

                self.clustered_img[i][j] = np.argmin(pixel_distances_to_means)
                self.error += np.min(pixel_distances_to_means)
        self.eps = np.abs(self.error - prev_error) / self.error

        print("\n", self.error, sep="")
        print("Image reclusterd")

        self.calculate_cluster_centers()

    @staticmethod
    def distance(a, b):
        if a is np.NAN or b is np.NAN:
             print(a, b)
        return np.linalg.norm(a-b)

    def show(self, option=Render.BLACK_AND_WHITE):
        if option is Render.CLUSTERS_CENTERS:
            tmp = self.img.copy()
            for k in range(self.k):
                tmp[self.clustered_img == k] = self.cluster_centers[k]
        elif option is Render.BLACK_AND_WHITE:
            tmp = self.clustered_img.copy()
            tmp = tmp.astype("uint8") * (255//self.k)

        else:
            tmp = self.img.copy()
            for k in range(self.k):
                tmp[self.clustered_img == k] = k_to_rgb(k, self.k)

        add_note_on_the_picture(tmp, "Eps: " + str(round(self.eps, 5)), label_center=(0, 0))
        return tmp


def show_exercise_2(k=3, transform=Filter.nothing, render=Render.CLUSTERS_CENTERS):
    """
    K-Means for color quantization

    :return: None
    """
    Lenna = cv2.imread('../Data/Lenna.png')
    Lenna = cv2.resize(Lenna, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    Lenna = transform(Lenna, label=False)
    segmented_lenna = Kmeans(Lenna, k=k, break_eps=0.001)

    while cv2.waitKey(100) != KEYS.SPACE:
        cv2.imshow('Exercise 2: K-Means', segmented_lenna.show(option=render))
        segmented_lenna.recluster()


def display_texts(img: ndarray, texts: list):
    from lib.common import FONT, FONT_SCALE, LINE_TYPE
    (label_width, label_height), _ = cv2.getTextSize(texts[0], FONT, FONT_SCALE, LINE_TYPE + 2)
    cimg = img.copy()
    x = (cimg.shape[0] - label_width)//2
    for i, text in enumerate(texts):
        y = (label_height*1.5)*i
        add_note_on_the_picture(cimg, text, label_center=(x, int(y)))

    return cimg


if __name__ == "__main__":

    blank_image = np.zeros((500, 500, 3), np.uint8)

    expected_colours = {KEYS.ONE:   Filter.nothing,
                        KEYS.TWO:   Filter.to_hsv,
                        KEYS.THREE: Filter.to_lab}

    expected_clusters = {KEYS.TWO:   2,
                         KEYS.THREE: 3,
                         KEYS.FOUR:  4,
                         KEYS.FIVE:  5,
                         KEYS.SIX:   6}

    expected_renders = {KEYS.ONE:   Render.BLACK_AND_WHITE,
                        KEYS.TWO:   Render.BRIGHT_COLOURS,
                        KEYS.THREE: Render.CLUSTERS_CENTERS}
    transform_ = False
    render_ = False
    k_ = False

    def choose_render():
        while (pressed_key := cv2.waitKey(10)) != KEYS.Q:
            cv2.imshow('Exercise 2: K-Means',
                       display_texts(blank_image,
                                     ["Choose Render          ",
                                      "Key 1 - Black and White",
                                      "Key 2 - Bright Colours",
                                      "Key 3 - Center Means"]))
            if pressed_key in expected_renders:
                return expected_renders[pressed_key]

    def choose_k_and_render():
        while (pressed_key := cv2.waitKey(10)) != KEYS.Q:
            cv2.imshow('Exercise 2: K-Means',
                       display_texts(blank_image,
                                     ["Choose the number of clusters",
                                      "Key 2 - 2 clusters",
                                      "Key 3 - 3 clusters",
                                      "Key 4 - 4 clusters",
                                      "Key 5 - 5 clusters",
                                      "Key 6 - 6 clusters"]))
            if pressed_key in expected_clusters:
                return expected_clusters[pressed_key], choose_render()

    while (pressed_key := cv2.waitKey(10)) != KEYS.Q:
        cv2.imshow('Exercise 2: K-Means',
                   display_texts(blank_image,
                                 ["Choose your Colour Space:",
                                  "Key 1 - RGB", "Key 2 - HSV", "Key 3 - LAB"]))
        if pressed_key in expected_colours:
            transform_ = expected_colours[pressed_key]

            k_, render_ = choose_k_and_render()

            show_exercise_2(k_, transform_, render_)

    cv2.destroyAllWindows()
