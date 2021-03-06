import cv2
import glob

import numpy as np

from operator import add
from functools import reduce


# TODO:
# 1. check if provided images have all the same size (w and h)
# 2. used common w and h to define keypoints

class BasicRetrieval:

    def __init__(self, path_to_trainset: str, path_to_testset: str, keypoints: tuple, pace=11):
        self.__trainset, self.__trainpath = self._load_images(path_to_trainset)
        self.__testset,  self.__testpath  = self._load_images(path_to_testset)

        self.__keypoints = None
        self.__train_desc, self.__test_desc = None, None
        self.__i_dist = None

        self.__create_keypoints(keypoints[0], keypoints[1], pace)

    def __create_keypoints(self, h, w, pace):
        self.__keypoints = [[cv2.KeyPoint(i*pace, j*pace, pace)
                             for j in range(1, w//pace)]
                            for i in range(1, h//pace)]

        self.__keypoints = reduce(add, self.__keypoints)  # kind of flattering

    @staticmethod
    def _load_images(path_to_set: str):
        image_pathes = glob.glob(path_to_set)
        return zip(*[(cv2.imread(path, cv2.IMREAD_COLOR), path) for path in image_pathes])

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

    @staticmethod
    def _half(img, sf=0.5):
        return cv2.resize(img, None, fx=sf, fy=sf, interpolation=cv2.INTER_AREA)

    def show(self, saveto="1_result/output.png"):

        tns = self.__trainset
        tts = self.__testset

        def combine_images(test_i):
            ii = self.__i_dist[test_i]

            uppper_ten = [self._half(tns[i]) for i in ii[:10]]
            lower_ten  = [self._half(tns[i]) for i in ii[10:20]]

            uppper_ten = cv2.hconcat(uppper_ten)
            lower_ten  = cv2.hconcat(lower_ten)

            train = cv2.vconcat([uppper_ten, lower_ten])
            return cv2.hconcat([tts[test_i], train])

        vis = cv2.vconcat([combine_images(i) for i in range(len(tts))])
        cv2.imshow("Matches", vis)
        cv2.imwrite(saveto, vis)
        cv2.waitKey(0)
