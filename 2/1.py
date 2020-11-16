import cv2
import glob

import numpy as np

from operator import add
from functools import reduce

# TODO:
# 1. check if provided images have all the same size (w and h)
# 2. used common w and h to define keypoints


class ImageRetrieval:

    def __init__(self, path_to_trainset: str, path_to_testset: str, keypoints: tuple):
        self.__trainset = self._load_images(path_to_trainset)
        self.__testset  = self._load_images(path_to_testset)

        self.__keypoints = None
        self.__train_desc, self.__test_desc = None, None
        self.__i_dist = None

        self.__create_keypoints(keypoints[0], keypoints[1])

    def __create_keypoints(self, h, w, pace=11):
        self.__keypoints = [[cv2.KeyPoint(i*pace, j*pace, pace)
                             for j in range(1, w//pace)]
                            for i in range(1, h//pace)]

        self.__keypoints = reduce(add, self.__keypoints)  # kind of flattering

    @staticmethod
    def _load_images(path_to_set: str):
        train_images = glob.glob(path_to_set)
        return [cv2.imread(path, cv2.IMREAD_COLOR) for path in train_images]

    @staticmethod
    def _distance(a, b):
        return np.linalg.norm(a-b)

    def compute_descriptors(self):  # train
        test  = self.__testset
        train = self.__trainset
        kpoints = self.__keypoints

        grayed = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()

        self.__train_desc = [sift.compute(grayed(img), kpoints)[1] for img in train]
        self.__test_desc  = [sift.compute(grayed(img), kpoints)[1] for img in test]

    def compare_sets(self):
        # as I am trying not to exceed 100 columns in a line, I had to do mapping
        test = self.__test_desc
        train = self.__train_desc
        d = self._distance

        self.__i_dist = [np.argsort([d(test_d, train_d) for train_d in train]) for test_d in test]

    def show(self):

        tns = self.__trainset
        tts = self.__testset

        half = lambda img, sf=0.5: cv2.resize(img, None, fx=sf, fy=sf, interpolation=cv2.INTER_AREA)

        def combine_images(test_i):
            ii = self.__i_dist[test_i]

            uppper_ten = [half(tns[i]) for i in ii[:10]]
            lower_ten  = [half(tns[i]) for i in ii[10:20]]

            uppper_ten = cv2.hconcat(uppper_ten)
            lower_ten  = cv2.hconcat(lower_ten)

            train = cv2.vconcat([uppper_ten, lower_ten])
            return cv2.hconcat([tts[test_i], train])

        vis = cv2.vconcat([combine_images(i) for i in range(len(tts))])
        cv2.imshow("Matches", vis)
        cv2.imwrite("1_result/output.png", vis)
        cv2.waitKey(0)


if __name__ == "__main__":
    foo = ImageRetrieval(path_to_trainset='../Data/images/db/*/*/*.jpg',
                         path_to_testset ='../Data/images/db/*/*.jpg',
                         keypoints=(256, 256))

    foo.compute_descriptors()
    foo.compare_sets()
    foo.show()

