from pandas.io.excel import read_excel
from matrix_factorization import PMF
import matplotlib.pyplot as plt
from utils import download
from scipy import sparse
import numpy as np
import zipfile
import os

"""
A dataset of jokes and associated user ratings.
Eigentaste: A Constant Time Collaborative Filtering Algorithm.
Ken Goldberg, Theresa Roeder, Dhruv Gupta, and Chris Perkins.
Information Retrieval, 4(2), 133-151. July 2001.
http://eigentaste.berkeley.edu/dataset/

We will use this dataset to test our simple recommendation algorithm.
"""
dataset_url = "http://eigentaste.berkeley.edu/dataset/jester_dataset_1_1.zip"

"""
Next, download the jester dataset to jester_dataset_1_1.zip if it hasn't been
downloaded yet
"""
dataset_fname = dataset_url.split("/")[-1]
if not os.path.exists(dataset_fname):
    download(dataset_url, server_fname=dataset_fname)

"""
The dataset is stored as an Excel spreadsheet (XLS).
We can read it without unzipping using the zipfile library.
"""
archive = zipfile.ZipFile(dataset_fname, 'r')
# Only one file in the zipfile we are reading from
# archive.open returns a file-like object - perfect for sending to pandas
file_handle = archive.open(archive.infolist()[0])

"""
To read the actual XLS file, we can use pandas.
"""
dataframe = read_excel(file_handle)
data = dataframe.values

"""
Only use the first 100 users for this example.
"""
user_indices = data[:100, 0]
ratings = data[:100, 1:]
# Necessary because this is a view of the underlying data, want separate copy
true_ratings = np.copy(data[:100, 1:])

"""
In this dataset, any rating of 99. means that a joke was unrated. Since these
are floating point values, it is best to create the sparse array by hand.
We can get these indices with np.where.
"""
rated = np.where(ratings <= 10.)
np.random.RandomState(1999)
# Use 20% for validation
n_validation = int(0.2 * len(rated[0]))
idx = np.random.randint(0, len(rated[0]), n_validation)
# Stack and transpose to get an (x, 2) array of indices
validation_indices = np.vstack((rated[0], rated[1])).T[idx]
# Set validation to NaN now
ratings[validation_indices[:, 0], validation_indices[:, 1]] = 99.
# Keep this mask for plotting to include validation
mask = (ratings <= 10.)
# Redo NaN check
rated = np.where(ratings <= 10.)
ratings = sparse.coo_matrix((ratings[rated[0], rated[1]], (rated[0], rated[1])))

"""
For now, treat this algorithm as a black box with input ratings,
output recommendation basis matrices which can be used for predictions.
If curious, see the docstrings of the function for the original paper.
"""
U, V, m = PMF(ratings, minibatch_size=10, learning_rate=0.001, momentum=0.95,
              regularization=0.75, max_epoch=100, rank=20, random_state=2000)
predicted_ratings = np.dot(U, V.T) + m
predicted_ratings = np.clip(predicted_ratings, -10, 10)

"""
Calculate mean absolute error on validation indices
"""
val_truth = true_ratings[validation_indices[:, 0], validation_indices[:, 1]]
val_predict = predicted_ratings[validation_indices[:, 0],
                                validation_indices[:, 1]]
# Mean absolute error
mae = np.mean(np.abs(val_truth - val_predict))
print("Validation mean absolute error %f" % mae)

"""
Plot the first 100 user prediction matrix
"""
f, axarr = plt.subplots(1, 2)
axarr[0].matshow((true_ratings * mask), cmap="gray")
axarr[0].set_title("Ground truth ratings")
axarr[1].matshow((predicted_ratings * mask), cmap="gray")
axarr[1].set_title("Predicted ratings\n Validation mean absolute error %f" % mae)
axarr[0].axis("off")
axarr[1].axis("off")
plt.show()
