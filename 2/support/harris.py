import cv2
import numpy as np


# Load image and convert to gray and floating point
img = cv2.imread('../Data/images/Lenna.png')
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
gray = np.float32(gray)

# Define sobel filter and use cv2.filter2D to filter the grayscale image

# YOUR CODE HERE

# Compute G_xx, G_yy, G_xy and sum over all G_xx etc. 3x3 neighbors to compute
# entries of the matrix M = \sum_{3x3} [ G_xx Gxy; Gxy Gyy ]
# Note1: this results again in 3 images sumGxx, sumGyy, sumGxy
# Hint: to sum the neighbor values you can again use cv2.filter2D to do this efficiently

# YOUR CODE HERE

# Define parameter
k = 0.04
threshold = 0.01

# Compute the determinat and trace of M using sumGxx, sumGyy, sumGxy. With det(M) and trace(M)
# you can compute the resulting image containing the harris corner responses with
# harris = ...

# YOUR CODE HERE

# Filter the harris 'image' with 'harris > threshold*harris.max()'
# this will give you the indices where values are above the threshold.
# These are the corner pixel you want to use

# YOUR CODE HERE

# The OpenCV implementation looks like this - please do not change
harris_cv = cv2.cornerHarris(gray,3,3,k)

# intialize in black - set pixels with corners in with
harris_cv_thres = np.zeros(harris_cv.shape)
harris_cv_thres[harris_cv>threshold*harris_cv.max()]=[255]

# just for debugging to create such an image as seen
# in the assignment figure.
# img[harris>threshold*harris.max()]=[255,0,0]


# please leave this - adjust variable name if desired
print("====================================")
print("DIFF:", np.sum(np.absolute(harris_thres - harris_cv_thres)))
print("====================================")


# cv2.imwrite("Harris_own.png", harris_thres)
# cv2.imwrite("Harris_cv.png", harris_cv_thres)
# cv2.imwrite("Image_with_Harris.png", img)

# cv2.namedWindow('Interactive Systems: Harris Corner')
# while True:
#     ch = cv2.waitKey(0)
#     if ch == 27:
#         break
#
#     cv2.imshow('harris',harris_thres)
#     cv2.imshow('harris_cv',harris_cv_thres)
