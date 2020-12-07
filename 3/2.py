
import numpy as np
import matplotlib.pyplot as plt

from lib.filter import Filter
from lib.neural_network import DataSet
from lib.neural_network import NeuralNetwork
from lib.neural_network import LossFunction as LF
from lib.neural_network import ActivationFunction as AF


def not_a_realistic_use_case():
    # Initial Settings =====
    np.random.seed(0)
    nn_img_size = 32
    num_classes = 3
    learning_rate = 0.0001
    num_epochs = 500
    batch_size = 4

    trainset = DataSet.case(path='../Data/images/db/train/*/*.jpg', pattern='(?<=train\/)(.*?)(?=\/)')
    testset  = DataSet.case(path='../Data/images/db/test/*.jpg',    pattern='(?<=test\/)(.*?)(?=.jpg)')

    # Prepare Dataset =====

    def reduce_normalize(img):
        img = Filter.reduce_size(img.astype(np.float64), target_size=(32, 32)).flatten()

        mean = np.mean(img)
        std  = np.std(img)
        return (img - mean)/std

    dataset = DataSet(trainset, testset, transform=reduce_normalize)

    X_train, Y_train = dataset.get_train()
    X_test,  Y_test  = dataset.get_test()

    # Train and Test =====
    np.random.seed(1)
    two_layer_nn_mse = NeuralNetwork(nn_structure=[nn_img_size ** 2, num_classes],
                                     f =[AF.relu           ],
                                     df=[AF.relu_derivative],
                                     J =LF.loss_mse,
                                     dJ=LF.loss_deriv_mse)

    two_layer_nn_ce = NeuralNetwork(nn_structure=[nn_img_size ** 2, num_classes],
                                     f =[AF.relu           ],
                                     df=[AF.relu_derivative],
                                     J =LF.loss_crossentropy,
                                     dJ=LF.loss_deriv_crossentropy)

    mse_loss = two_layer_nn_mse.train(X_train, Y_train, n_epoch=num_epochs, batch_size=batch_size, alpha=learning_rate)
    ce_loss = two_layer_nn_ce.train(X_train, Y_train, n_epoch=num_epochs, batch_size=batch_size, alpha=learning_rate)

    plt.subplot(1, 2, 1)
    plt.plot(mse_loss)
    plt.title('MSE Error')
    plt.xlabel('Epoch number')
    plt.ylabel('Average J   ')

    plt.subplot(1, 2, 2)
    plt.plot(ce_loss)
    plt.title('Cross Entropy')
    plt.xlabel('Epoch number')
    plt.ylabel('Average J   ')
    plt.savefig("2_results/two_layer_nn_train__loss_function_comparison.png")
    plt.show()

    print("MSE Accuracy:", two_layer_nn_mse.predict(X_test, Y_test), "%")
    print("CE Accuracy:", two_layer_nn_ce.predict(X_test, Y_test), "%")

    print("------------------------------------")
    print("Test model output MSE Weights:", two_layer_nn_mse.get_weights())
    print("Test model output MSE Bias:",    two_layer_nn_mse.get_bias())
    print("------------------------------------")
    print("Test model output CE Weights:", two_layer_nn_ce.get_weights())
    print("Test model output CE Bias:",    two_layer_nn_ce.get_bias())


def a_realistic_use_case():
    # === MNIST example
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    X_scale = StandardScaler()
    X = X_scale.fit_transform(digits.data)

    y = digits.target
    np.random.seed(1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    def to_vect(y):
        y_vect = np.zeros((len(y), 10))
        for i in range(len(y)):
            y_vect[i, y[i]] = 1
        return y_vect

    y_train = to_vect(y_train)
    y_test = to_vect(y_test)

    three_layer_nn_sse = NeuralNetwork(nn_structure=[64, 30, 10],
                           f =[AF.sigmoid,            AF.sigmoid],
                           df=[AF.sigmoid_derivative, AF.sigmoid_derivative],
                           J =LF.loss_sse,
                           dJ=LF.loss_deriv_sse)

    loss = three_layer_nn_sse.train(X_train, y_train, n_epoch=3000, batch_size=len(y_train), alpha=0.25)
    plt.plot(loss)
    plt.title('SSE Error')
    plt.xlabel('Iteration number')
    plt.ylabel('J')
    plt.savefig("2_results/three_layer_nn_train__real_case.png")
    plt.show()

    print("SSE Accuracy:", three_layer_nn_sse.predict(X_test, y_test), "%")

    print("------------------------------------")
    print("Test model output SSE Weights:", three_layer_nn_sse.get_weights())
    print("Test model output SSE Bias:",    three_layer_nn_sse.get_bias())


not_a_realistic_use_case()
a_realistic_use_case()
