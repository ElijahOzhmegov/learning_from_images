import numpy as np
import cv2
import math
import sys

from lib.common import KEYS
from lib.common import add_note_on_the_picture
from lib.common import k_to_rgb
from enum import Enum


class Render(Enum):
    CLUSTERS_CENTERS = 0
    BLACK_AND_WHITE = 1
    BRIGHT_COLOURS = 2


class Kmeans():

    def __init__(self, img, k=3, break_eps=.001):
        self.k = k
        self.img = img.copy()

        self.n_layers = img.shape[-1]
        self.cluster_centers = np.zeros((k, img.shape[-1]))
        self.clustered_img = np.random.randint(k, size=img.shape[:2])

        self.break_eps = break_eps
        self.error = 0
        self.eps = 1.

        print("Initialised")
        self.calculate_cluster_centers()

    def calculate_cluster_centers(self):

        for i in range(self.k):
            k_elements = self.clustered_img == i
            for j in range(self.n_layers):
                L = self.img[k_elements, j]
                self.cluster_centers[i, j] = np.mean(L)

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


def show_exercise_2():
    """
    K-Means for color quantization

    :return: None
    """
    Lenna = cv2.imread('../Data/Lenna.png')
    segmented_lenna = Kmeans(Lenna, k=8)

    while cv2.waitKey(100) != KEYS.Q:
        cv2.imshow('Exercise 2: K-Means', segmented_lenna.show(option=Render.BRIGHT_COLOURS))
        segmented_lenna.recluster()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_exercise_2()
