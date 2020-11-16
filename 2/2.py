import cv2

import matplotlib.pyplot as plt
import numpy as np

from lib.common import load_images


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

    def show(self, i: int = 0, label: str = None, show=True):
        n = self.__nbucket
        i = i % n  # fool protection, i is number of the KeyPoint
        feature = self.features[i]

        plt.bar(np.arange(n), feature, align='edge')
        # plt.grid(True)
        plt.xticks(range(n), (str(j*(360//n)) + '°' for j in range(n)))
        # plt.tick_params(axis="x", direction="out", pad=22)

        if label: plt.title(label + ': feature ' + str(i))
        if show: plt.show()

    def save(self, i: int = 0, label: str = None):
        self.show(i, label, show=False)
        plt.savefig('2_result/' + label.split('/')[-1])
        plt.close()


if __name__ == '__main__':

    img_paths = load_images('../Data/images/hog_test/*.jpg')
    keypoints = [cv2.KeyPoint(15, 15, 11)]

    for img_path in img_paths:
        img, path = img_path

        foo = LocalFeatureExtractor(img, keypoints)
        foo.extract_features()
        foo.save(label=str(path))
