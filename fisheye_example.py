import cv2
import numpy as np

assert float(cv2.__version__.rsplit('.', 1)[0]) >= 3, 'OpenCV version 3 or newer required.'

K = np.array([[  689.21,     0.  ,  1295.56],
              [    0.  ,   690.48,   942.17],
              [    0.  ,     0.  ,     1.  ]])

# zero distortion coefficients work well for this image
D = np.array([0., 0., 0., 0.])

# use Knew to scale the output
Knew = K.copy()
Knew[(0,1), (0,1)] = 0.4 * Knew[(0,1), (0,1)]


img = cv2.imread('fisheye_sample.jpg')
img_undistorted = cv2.fisheye.undistortImage(img, K, D=D, Knew=Knew)
cv2.imwrite('fisheye_sample_undistorted.jpg', img_undistorted)
cv2.imshow('undistorted', img_undistorted)
cv2.waitKey()

