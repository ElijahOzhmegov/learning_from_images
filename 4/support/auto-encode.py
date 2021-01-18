import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from torch.autograd import Variable


def load_images(path: str, file_ending: str = ".png") -> np.ndarray:
    """
    Load all images in path with matplotlib that have given file_ending

    Arguments:
        path: path of directory containing image files that can be assumed to have all the same dimensions
        file_ending: string that image files have to end with, if not->ignore file

    Return:
        images: A 3-D Numpy Array representing multiple images
                Dim 1 = Number of images
                Dim 2 = Height of images
                Dim 3 = Width of images
    """

    images = []

    files = os.listdir(path)
    files.sort()
    for cur in files:
        if not cur.endswith(file_ending):
            continue

        try:
            image = mpl.image.imread(path + cur)
            img_mtx = np.asarray(image, dtype="float64")
            images.append(img_mtx)
        except:
            continue

    return np.array(images)


class Autoencoder(nn.Module):
    # n_pixels: The number of pixels in the image
    # feature_length: Vector length of the encoded image
    def __init__(self, n_pixels, feature_length):
        super(Autoencoder, self).__init__()
        # 1.1 Define Encoder layer
        # 1.2 Define decoder layer
        # TODO YOUR CODE HERE

    def forward(self, x):
        # 1.3 Define forward path
        # TODO YOUR CODE HERE
        return x


if __name__ == '__main__':

    images, x, y = load_images('./data/train/')

    # Load images
    images = load_images('./data/train/')
    y, x = images.shape[1:3]
    # Flatten last two dimensions by reshaping the array
    images = images.reshape(images.shape[0], x * y)

    # 2.1 Calculate mean values for each pixel across all images
    # 2.2 Subtract mean values from images to center the data
    mean = images.mean(axis=0)
    images[:] -= mean

    # Set Hyperparameters / Please change as needed
    num_epochs = 250
    batch_size = 50
    learning_rate = 0.01

    data = torch.from_numpy(images)

    # Load Autoencoder model with number of pixels in our images
    model = Autoencoder(n_pixels=x * y, feature_length=300)
    # Define Loss
    criterion = nn.MSELoss()
    # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-05)

    prev_loss = 10000.0
    best_model = None

    for epoch in range(num_epochs):
        data = Variable(data.float())
        # ===================forward=====================
        output = model(data)
        loss = criterion(output, data)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Save best model
        if loss.item() < prev_loss:
            print(f'saved best model with loss: {loss.item()}')
            best_model = model
            prev_loss = loss.item()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'.format(
            epoch + 1, num_epochs, loss.item(), loss.item()))

    # Load, flatten and center the test images
    images_test = load_images('./data/test/')
    y, x = images_test.shape[1:3]

    # 3. Flatten and center test images (reshape and sub mean)
    # (Use the mean calculated from the training images)
    images_test_normalized = images_test.reshape(images_test.shape[0], x * y) - mean
    data_test = torch.from_numpy(images_test_normalized)

    # List for reconstructed images to plot it later
    reconstructed_images = []

    # Loop through all normalized test images (data_test)
    # to encode and reconstruct the test images with our neural network autoencoder
    # and measure the reconstruction error between reconstructed and original image
    errors = []
    for i, test_image in enumerate(images_test):
        print(f'----- image[{i}] -----')
        # 4.1 Encode and reconstruct image with best nn model
        pred = best_model(data_test[i, :].float())
        pred_np = pred.data.numpy()
        print(f'Prediction shape: {pred_np.shape}')
        
        # 4.2 add mean to pred_np pred_np and reshape to size (116,98) to reconstruct the image
        # (Use the mean calculated from the training images)
        pred_np += mean
        img_reconst = pred_np.reshape((y, x))
        reconstructed_images.append(img_reconst)
        print(f'Reconstructed mage shape: {img_reconst.shape}')
        # Measure error between loaded original image and reconstructed image
        error = np.linalg.norm(test_image - img_reconst)
        errors.append(error)
        print(f'Reconstruction error: {error}')

    # Plot Results
    if len(images_test) != 0 and len(reconstructed_images) != 0:
        plot_img_original = images_test[-1]
        plot_img_reconstructed = reconstructed_images[-1]

        grid = plt.GridSpec(2, 9)

        plt.subplot(grid[0, 0:3])
        plt.imshow(plot_img_original, cmap='Greys_r')
        plt.xlabel('Original image')

        plt.subplot(grid[0, 3:6])
        plt.imshow(plot_img_reconstructed, cmap='Greys_r')
        plt.xlabel('Reconstructed image')

        plt.subplot(grid[0, 6:])
        plt.plot(np.arange(len(images_test)), errors)
        plt.xlabel('Errors all images')

        print("Mean error", np.asarray(errors).mean())

        plt.savefig("pca_ae_solution.png")
        plt.show()
    else:
        print(
            'Make sure to fill image_test and reconstructed_images lists with images to show.'
        )
