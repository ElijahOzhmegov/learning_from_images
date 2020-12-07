import cv2
import numpy as np
import matplotlib.pyplot as plt
from lib.neural_network import DataSet
from lib.neural_network import NeuralNetwork
from lib.neural_network import LossFunction as LF
from lib.neural_network import ActivationFunction as AF
from lib.filter import Filter

np.random.seed(0)
nn_img_size = 32
num_classes = 3
learning_rate = 0.0001
num_epochs = 500
batch_size = 4

trainset = DataSet.case(path='../Data/images/db/train/*/*.jpg', pattern='(?<=train\/)(.*?)(?=\/)')
testset  = DataSet.case(path='../Data/images/db/test/*.jpg',    pattern='(?<=test\/)(.*?)(?=.jpg)')


def reduce_normalize(img):
    img = Filter.reduce_size(img.astype(np.float32), target_size=(32, 32))

    mean = np.mean(img)
    std  = np.std(img)
    return (img - mean)/std


dataset = DataSet(trainset, testset, transform=reduce_normalize)

X_train, Y_train = dataset.get_train()
X_test,  Y_test  = dataset.get_test()

# plt.gray()
# plt.matshow(X_train[0])
# plt.show()
#
# print(Y_train)


# =====

# foobar = NeuralNetwork(nn_structure=[nn_img_size ** 2, 81, num_classes],
#                        f =[AF.relu,            AF.softmax],
#                        df=[AF.relu_derivative, AF.softmax_derivative],
#                        J =LF.loss_mse,
#                        dJ=LF.loss_deriv_mse)
#
# loss = foobar.train(X_train, Y_train, n_epoch=100, batch_size=batch_size, alpha=learning_rate)
#
# plt.plot(loss)
# plt.ylabel('J')
# plt.xlabel('Iteration number')
# plt.show()
#
# evaluation = foobar.predict(X_test, Y_test)
# print(evaluation)

# === MNIST example
from sklearn.datasets import load_digits
digits = load_digits()

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
X_scale = StandardScaler()
X = X_scale.fit_transform(digits.data)

from sklearn.model_selection import train_test_split
y = digits.target
np.random.seed(1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 10))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect
y_v_train = convert_y_to_vect(y_train)
y_v_test = convert_y_to_vect(y_test)

nn_structure = [64, 30, 10]


def tmp_loss(y, h):
    return np.linalg.norm((y-h))


foobar = NeuralNetwork(nn_structure=nn_structure,
                       f =[AF.sigmoid,            AF.sigmoid],
                       df=[AF.sigmoid_derivative, AF.sigmoid_derivative],
                       J =tmp_loss,
                       dJ=LF.loss_deriv_sse)
Y = convert_y_to_vect(y)
m = len(Y)
loss = foobar.train(X, Y, n_epoch=100, batch_size=m, alpha=0.25)
plt.plot(loss)
plt.ylabel('J')
plt.xlabel('Iteration number')
plt.show()
