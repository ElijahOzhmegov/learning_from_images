import re
import glob
import collections
import cv2
import numpy as np
from difflib import SequenceMatcher
import numpy.random as r


class DataSet:
    case = collections.namedtuple("DataSet", "path pattern")

    def __init__(self, trainset: case, testset: case,
                 colorspace=cv2.IMREAD_GRAYSCALE,
                 transform=lambda img: img):
        self.trainset = self._load(trainset, colorspace, transform)
        self.testset  = self._load(testset,  colorspace, transform)

        self.__y_train, self.__y_test = None, None
        self.__convert_outputs_to_numbers()


    @staticmethod
    def _load(dataset, colorspace=cv2.IMREAD_GRAYSCALE, transform=lambda img: img):
        image_paths = glob.glob(dataset.path)
        getname = lambda line: re.search(dataset.pattern, line).group(0)
        getimg  = lambda path: transform(cv2.imread(path, colorspace))
        return [(getimg(path), getname(path)) for path in image_paths]

    def __convert_outputs_to_numbers(self):
        _, testoutput  = zip(*self.testset)
        _, trainoutput = zip(*self.trainset)

        words      = testoutput
        dictionary = set(trainoutput)

        def similar(a, b): return SequenceMatcher(None, a, b).ratio()

        word_to_number = dict(zip(dictionary, range(len(dictionary))))
        to_dictionary = {new_word:
                             max({similar(new_word, old_word): old_word for old_word in dictionary}.items())[1]
                         for new_word in words}

        y_test  = np.array([word_to_number[to_dictionary[word]] for word in  testoutput])
        y_train = np.array([word_to_number[word]                for word in trainoutput])

        def create_output_matrix(y_array):
            n = len(y_array)
            m = len(dictionary)

            matrix = np.zeros((n, m))
            matrix[(list(range(n)), y_array)] = 1

            return matrix

        self.__y_train = create_output_matrix(y_train)
        self.__y_test  = create_output_matrix(y_test )

    def get_train(self):
        x_output, _ = zip(*self.trainset)
        return x_output, self.__y_train

    def get_test(self):
        x_output, _ = zip(*self.testset)
        return x_output, self.__y_test


class NeuralNetwork:

    @staticmethod
    def _list_to_dict(l: list):
        return {key + 2: value for key, value in enumerate(l)}

    def __init__(self, nn_structure: list, f: list, df: list, J, dJ):
        self.__nn_structure = nn_structure
        self.__f  = self._list_to_dict( f)
        self.__df = self._list_to_dict(df)
        self.__J  =  J
        self.__dJ = dJ

        self.__W = {}
        self.__B = {}
        self.__init_weights()

        self.__delta = {}
        self.__Delta_W = {}
        self.__Delta_B = {}
        self.__init_deltas()

        self.__loss = []

    def __init_weights(self):
        np.random.seed(123)
        n_str = self.__nn_structure
        nl = len(n_str)
        for ll in range(1, nl):
            self.__W[ll] = r.random_sample((n_str[ll], n_str[ll - 1]))
            self.__B[ll] = r.random_sample((n_str[ll],))

        print("weights are initialized")

    def __init_deltas(self):
        n_str = self.__nn_structure
        for ll in range(1, len(n_str)):
            self.__Delta_W[ll] = np.zeros((n_str[ll], n_str[ll - 1]))
            self.__Delta_B[ll] = np.zeros((n_str[ll],))

    def __feed_forward(self, x):
        W = self.__W
        B = self.__B
        f = self.__f

        # if it is the first layer, then the input into the weights is x, otherwise,
        # it is the output from the last layer
        h = {1: x}
        z = {}
        for ll in range(1, len(W) + 1):
            # $ z^{(l+1)} = W^{(l)}*h^{(l)} + b^{(l)} $
            z[ll + 1] = W[ll].dot(h[ll]) + B[ll]
            # $ h^{(l)} = f(z^{(l)}) $
            h[ll + 1] = f[ll + 1](z[ll + 1])
        return h, z

    def __calculate_out_layer_delta(self, y, h_out, z_out):
        nl = len(self.__nn_structure)
        df = self.__df[nl]
        dJ = self.__dJ
        # $ delta^{(nl)} = dJ(y_i, h_i^{(nl)}) * f'(z_i^{(nl)}) $
        delta = dJ(y, h_out) * df(z_out)
        self.__delta[nl] = delta
        return self.__delta[nl]

    def __calculate_hidden_layer_delta(self, ll, z_l):
        df = self.__df[ll]
        w_l = self.__W[ll]
        delta_l_plus_1 = self.__delta[ll+1]
        # $ delta^{(l)} = (transpose(W^{(l)}) * delta^{(l+1)}) * f'(z^{(l)}) $
        delta = np.dot(np.transpose(w_l), delta_l_plus_1) * df(z_l)
        self.__delta[ll] = delta
        return self.__delta[ll]

    def __calculate_loss(self, y, h):
        J = self.__J
        self.__loss.append(np.abs(J(y, h)))

    def __backpropagation(self, y, h, z):
        nl = len(self.__nn_structure)

        h_out = h[nl]
        z_out = z[nl]
        self.__calculate_out_layer_delta(y, h_out, z_out)
        self.__calculate_loss(y, h_out)

        for ll in range(nl - 1, 1, -1):
            self.__calculate_hidden_layer_delta(ll, z[ll])

        return self.__delta

    def __update_deltas(self, h, delta):
        nl = len(self.__nn_structure)

        for ll in range(nl - 1, 0, -1):
            # $ \Delta W^{(l)} = \Delta W^{(l)} + delta^{(l+1)} * transpose(h^{(l)}) $
            self.__Delta_W[ll] += np.dot(delta[ll + 1][:, np.newaxis], np.transpose(h[ll][:, np.newaxis]))
            # $ \Delta B^{(l)} = \Delta B^{(l)} + delta^{(l+1)} $
            self.__Delta_B[ll] += delta[ll + 1]

    def __gradient_descent_step(self, alpha):
        nl = len(self.__nn_structure)
        m = self.__nn_structure[-1]

        Delta_W = self.__Delta_W
        Delta_B = self.__Delta_B

        for l in range(nl - 1, 0, -1):
            self.__W[l] += -alpha * (1.0 / m * Delta_W[l])
            self.__B[l] += -alpha * (1.0 / m * Delta_B[l])

    @staticmethod
    def _average(delta):
        bs = len(delta)  # batch size
        layers = delta[0].keys()

        output_delta = delta[0]
        for layer in layers: output_delta[layer] /= bs
        for ib in range(1, bs):
            for layer in layers: output_delta[layer] += delta[ib][layer]/bs

        return output_delta

    def __go_thru_trainset(self, X, Y, batch_size=4, alpha=0.25):
        set_size = len(Y)

        feed_forward = self.__feed_forward
        backpropagation = self.__backpropagation
        update_weights = self.__update_deltas
        average = self._average
        gradient_descent_step = self.__gradient_descent_step

        k = -1
        while (k := k + 1) < set_size:
            delta = dict()
            leap = np.min([k + batch_size, set_size])
            h = None
            for k in range(k, leap):
                h, z = feed_forward(X[k].flatten())
                delta[k % batch_size] = backpropagation(Y[k], h, z)

            averaged_delta = average(delta)
            update_weights(h, averaged_delta)
            gradient_descent_step(alpha)

            # print("Weights updated, k: ", k)

    def train(self, X, Y, n_epoch=80, batch_size=4, alpha=0.25):

        loss = []
        for i in range(n_epoch):
            print("Epoch:", i)
            self.__go_thru_trainset(X, Y, batch_size, alpha)
            loss.append(np.mean(self.__loss))
            self.__loss = []

        return loss

    def predict(self, X, Y):
        nl = len(self.__nn_structure)
        set_size = len(Y)

        feed_forward = self.__feed_forward
        loss = []

        for k in range(set_size):
            h, _ = feed_forward(X[k].flatten())
            correct = np.argmax(Y[k])
            predicted = np.argmax(h[nl])
            loss.append((correct, predicted))

        return loss


class ActivationFunction:

    @staticmethod
    def relu(z):
        """ReLU activation function"""
        return np.maximum(z, 0)

    @staticmethod
    def relu_derivative(z):
        """derivative of the ReLU activation function"""
        z[z <= 0] = 0
        z[z > 0] = 1
        return z

    @staticmethod
    def softmax(z):
        """softmax function to transform values to probabilities"""
        numerator = np.exp(z)
        denominator = np.sum(numerator)
        return numerator/denominator

    @staticmethod
    def softmax_derivative(z):
        numerator = np.exp(z)
        esum = np.sum(numerator)
        denominator = np.power(esum, 2)
        numerator = (esum - z) * z
        return numerator/denominator

    @staticmethod
    def stable_softmax(z):
        """stable softmax function to transform values to probabilities"""
        return ActivationFunction.softmax(z - z.max())

    @staticmethod
    def stable_softmax_derivative(z):
        return ActivationFunction.softmax_derivative(z)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        f = ActivationFunction.sigmoid
        return f(z) * (1 - f(z))


class LossFunction:

    @staticmethod
    def loss_sse(y, h):
        return np.power(y - h, 2).sum()/2.0

    @staticmethod
    def loss_deriv_sse(y, h):
        return -(y - h)

    @staticmethod
    def loss_mse(y, h):
        """mean squared loss function"""
        # use MSE error as loss function
        # Hint: the computed error needs to get normalized over
        #       the number of samples
        loss = np.power(y - h, 2).sum()/2.0
        mse = 1.0 / y.shape[0] * loss
        return mse

    @staticmethod
    def loss_deriv_mse(y, h):
        """derivative of the mean squared loss function"""
        dCda2 = -(1 / y.shape[0]) * (y - h)
        return dCda2

    @staticmethod
    def loss_crossentropy(activation, y_batch):
        """cross entropy loss function"""
        batch_size = y_batch.shape[0]
        loss = (-y_batch * np.log(activation)).sum() / batch_size
        return loss

    @staticmethod
    def loss_deriv_crossentropy(activation, y_batch):
        """derivative of the mean cross entropy loss function"""
        batch_size = y_batch.shape[0]
        dCda2 = activation
        dCda2[range(batch_size), np.argmax(y_batch, axis=1)] -= 1
        dCda2 /= batch_size
        return dCda2
