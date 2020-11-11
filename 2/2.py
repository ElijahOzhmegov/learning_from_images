import cv2
import math
import matplotlib
import glob

from lib.filter import Filter

import matplotlib.pyplot as plt
import numpy as np


class LocalFeatureExtractor:

    def __init__(self, img: np.ndarray, keypoints: list, nbucket=8):
        self.__img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.__keypoints = keypoints
        self.__nbucket = nbucket

        self.features = None

    def __extract_img_chunks(self):
        kps = self.__keypoints
        img = self.__img

        def extract_chunk(keypoint):

            radii = keypoint.size * 1.5

            lshift = radii // 2
            rshift = radii - lshift + 1

            pt = np.array(keypoint.pt)

            ui, lj = pt - lshift
            di, rj = pt + rshift

            # TODO: SIGABRT prevention

            return img[int(ui):int(di), int(lj):int(rj)]

        return [extract_chunk(kp) for kp in kps]

    def __calculate_chunk_feature(self, chunk, apply_mask=False):
        xsobel = cv2.Sobel(chunk, cv2.CV_32F, 1, 0, ksize=5)
        ysobel = cv2.Sobel(chunk, cv2.CV_32F, 0, 1, ksize=5)

        n = self.__nbucket
        divisor = 360 // n

        phase = (cv2.phase(xsobel, ysobel, angleInDegrees=True) // divisor).astype(int)
        magnitude = cv2.magnitude(xsobel, ysobel)

        if apply_mask:
            b, a = np.array(magnitude.shape) // 2
            mask = np.zeros(magnitude.shape)
            cv2.ellipse(mask, (a, b), (a, b), 0, 0, 360, 111, -1)

            magnitude[mask < 1] = 0

        buckets = np.array([np.sum(magnitude[phase == i]) for i in range(n)])
        buckets /= np.sum(buckets)

        return buckets

    def extract_features(self):
        chunks = self.__extract_img_chunks()
        calc = self.__calculate_chunk_feature

        self.features = [calc(chunk) for chunk in chunks]

    def show(self, i: int=0):
        n = self.__nbucket
        feature = self.features[i]
        plt.bar(np.arange(n), feature)
        plt.show()


def load_images(path_to_set: str):
    train_images = glob.glob(path_to_set)
    return [cv2.imread(path, cv2.IMREAD_COLOR) for path in train_images]


if __name__ == '__main__':

    imgs = load_images('../Data/images/hog_test/*.jpg')
    keypoints = [cv2.KeyPoint(15, 15, 11)]

    for img in imgs:
        foo = LocalFeatureExtractor(img, keypoints)
        foo.extract_features()
        foo.show()
