# Learning from images - Homeworks

Requirement: **Python 3.8**.

## Assignment 1

Folder `1` contains `1.py` and `2.py` that correspond to exercise 1.{1-4} and exercise 2.


### Exercise 1.1: 
  * just press `Space`

### Exercise 1.2: 
  * Key `1` - HSV
  * Key `2` - LAB
  * Key `3` - YUV
  * Key `4` - Adaptive Gausian Thresholding
  * Key `5` - OTSU' Thresholding
  * Key `6` - Canny Edge Detection
  * Key `7` - SIFT Detection
  * Key `9` - Gaussian Blur
  * Key `0` - Nothing
  * Key `q` - Quit
  
### Exercise 1.3: 
  * Key `q` - Quit
  
### Exercise 1.4:
  * Key `1` - OpenCV based Sobel edge detection
  * Key `2` - Fourier based Sobel edge detection (My smart approach)
  * Key `3` - For loop based Sobel edge detection (Stupid approach)
  * Key `q` - Quit
  
### Exercise 2
  * Key `Space` - Quit from viewing a clustered image
  * Key `q` - Quit
  * Keys `1-6` - to choose
  
#### a) What are the problems of this clustering algorithm? 
General problems of k-means clustering:
  1. Choosing k manually
  2. Dependence on the initial values (Main in current assignment)
  3. Lack of robustness (sensitivity to outliers)
  4. Distance is sensitive to the number of dimensions 
  
#### b) How can I improve the results?
Probable solutions:
  1. Scree plot
  2. Kmeans++ approach (Which was implicitly implemented)
  3. Use medians (even better PAM algorithm)
  4. Dimension reduction by using PCA (PCA is here as simple ans well-known  approach)
