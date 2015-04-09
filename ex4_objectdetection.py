import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from sklearn_theano.feature_extraction import OverfeatLocalizer
from sklearn_theano.feature_extraction import OverfeatTransformer

"""
pip install git+https://github.com/sklearn-theano/sklearn-theano

Example originally from sklearn-theano:
http://sklearn-theano.github.io/auto_examples/plot_localization_tutorial.html
"""


def convert_points_to_box(points, color, alpha, window_adjust=0):
    upper_left_point = (points[:, 0].min(), points[:, 1].min())
    width = points[:, 0].max() - points[:, 0].min()
    height = points[:, 1].max() - points[:, 1].min()
    return Rectangle(upper_left_point, width, height, ec=color,
                     fc=color, alpha=alpha)

# Show the original image
f, axarr = plt.subplots(2, 2)
# Data needs to be float32  and 0 - 255
X = mpimg.imread("sloth.png").astype('float32')
# Shrink the image so that processing is faster
X = imresize(X, .4)
# Zero pad with half of input size on each size
pad_size = 230 // 2
XL = np.zeros((2 * pad_size + X.shape[0], 2 * pad_size + X.shape[1], 3),
              dtype='float32')
XL[pad_size:pad_size + X.shape[0], pad_size:pad_size + X.shape[1], :] = X
X = XL
print("Read image")
print("Image size %sx%sx%s" % X.shape)
axarr[0, 0].imshow(X / 255.)
axarr[0, 0].axis('off')

# Show a single box
axarr[0, 1].imshow(X / 255.)
axarr[0, 1].axis('off')
r = Rectangle((0, 0), 231, 231, fc='yellow', ec='black', alpha=.8)
axarr[0, 1].add_patch(r)

# Show all the boxes being processed
axarr[1, 0].imshow(X / 255.)
axarr[1, 0].axis('off')
# Hard code box size to speed up processing
x_points = np.linspace(0, X.shape[1] - 231, 13)
y_points = np.linspace(0, X.shape[0] - 231, 10)
xx, yy = np.meshgrid(x_points, y_points)
for x, y in zip(xx.flat, yy.flat):
    axarr[1, 0].add_patch(Rectangle((x, y), 231, 231, fc='yellow', ec='black',
                          alpha=.4))

print("Starting localization")
# Get all points with sloth in the top 5 labels
sloth_label = "three-toed sloth, ai, Bradypus tridactylus"
clf = OverfeatLocalizer(match_strings=[sloth_label])
sloth_points = clf.predict(X)[0]
axarr[1, 1].imshow(X / 255.)
axarr[1, 1].axis('off')
axarr[1, 1].autoscale(enable=False)
axarr[1, 1].scatter(sloth_points[:, 0], sloth_points[:, 1], color='orange',
                    s=50)
print("Localization complete!")
plt.show()
