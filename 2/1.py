from lib.retrieval import BasicRetrieval


class ImageRetrieval(BasicRetrieval):
    pass


if __name__ == "__main__":
    foo = ImageRetrieval(path_to_trainset='../Data/images/db/*/*/*.jpg',
                         path_to_testset ='../Data/images/db/*/*.jpg',
                         keypoints=(256, 256))

    foo.compute_descriptors()
    foo.compare_sets()
    foo.show()

