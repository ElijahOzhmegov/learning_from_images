import re
import cv2
import numpy as np

from sklearn import svm
from difflib import SequenceMatcher

from lib.retrieval import BasicRetrieval
from lib.common import add_note_on_the_picture




class SVMImageRetrieval(BasicRetrieval):

    def __init__(self, trainset: tuple, testset: tuple,
                 keypoints: tuple, pace=15):
        super(SVMImageRetrieval, self).__init__(path_to_trainset=trainset[0],
                                                path_to_testset =testset[0],
                                                keypoints       =keypoints,
                                                pace            =pace)

        self.__train_cnames = self._load_classnames(self._BasicRetrieval__trainpath, trainset[1])
        self.__test_cnames  = self._load_classnames(self._BasicRetrieval__testpath,  testset[1])

        self.__x_train, self.__x_test = None, None
        self.__y_train, self.__y_test = None, None
        self.__predicted_y = None

        self.__clf = None

    @staticmethod
    def _load_classnames(pathlist: list, pattern: str):
        classname = lambda line: re.search(pattern, line).group(0)
        return [classname(path) for path in pathlist]

    @staticmethod
    def _similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    def __transform_target_variable(self):
        nameset = set(self.__train_cnames)
        name_to_int = dict(zip(nameset, range(len(nameset))))

        test_nameset = set(self.__test_cnames)

        to_nameset = {test_item: max({self._similar(test_item, item): item for item in nameset}.items())[1]
                      for test_item in test_nameset}

        self.__y_train = np.array([name_to_int[item] for item in self.__train_cnames])
        self.__y_test  = np.array([name_to_int[to_nameset[item]] for item in self.__test_cnames])

    def __train(self):
        X = self.__x_train
        Y = self.__y_train
        self.__clf = svm.SVC(decision_function_shape='ovo')
        self.__clf.fit(X, Y)

    def compute_descriptors(self):
        super(SVMImageRetrieval, self).compute_descriptors()

        flattering = lambda array:  np.array([np.array(img).flatten() for img in array])
        self.__x_train = flattering(self._BasicRetrieval__train_desc)
        self.__x_test  = flattering(self._BasicRetrieval__test_desc)

        self.__transform_target_variable()
        self.__train()

    def compare_sets(self):
        predicted_y = self.__clf.predict(self.__x_test)
        print("Accuracy", (np.sum(predicted_y == self.__y_test)/len(predicted_y))*100, '%')
        self.__predicted_y = predicted_y

    def show(self, saveto="1_results/output.png"):
        tns = np.array(self._BasicRetrieval__trainset)
        tts = np.array(self._BasicRetrieval__testset)

        y_tns = self.__y_train
        y_tts = self.__predicted_y

        half = super(SVMImageRetrieval, self)._half

        def concatenate(a_set, y_label, desired_label):
            pool = a_set[y_label == desired_label]
            pool = [half(img) for img in pool]
            return cv2.hconcat(pool)

        res = list()
        for ii, timg in enumerate(tts):
            # train = concatenate(tns, y_tns, y_tts[ii])
            add_note_on_the_picture(timg, self.__test_cnames[ii])
            pool = tns[y_tns == y_tts[[ii]]]
            pool = cv2.hconcat([half(img) for img in pool])

            # tmp = cv2.hconcat(half(timg))
            tmp = cv2.hconcat([half(timg), pool])
            res.append(tmp)

        for i, item in enumerate(res):
            cv2.imshow("PRESS ANY KEY " + str(i), item)
        # cv2.imwrite(saveto, vis)
        cv2.waitKey(0)


if __name__ == "__main__":
    foo = SVMImageRetrieval(trainset=('../Data/images/db/*/*/*.jpg',
                                      '(?<=train\/)(.*?)(?=\/)'),
                            testset=('../Data/images/db/*/*.jpg',
                                     '(?<=test\/)(.*?)(?=.jpg)'),
                            keypoints=(256, 256))

    foo.compute_descriptors()
    foo.compare_sets()
    foo.show()
