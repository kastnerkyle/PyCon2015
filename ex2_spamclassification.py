from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import BernoulliNB
from utils import download
import numpy as np
import zipfile
import os

"""
This example is modified from an excellent tutorial by Radim Rehurek, author of
gensim. http://radimrehurek.com/data_science_python/

The dataset we will be using is a bunch of texts, classed spam/not spam.
See https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
for more details.
"""
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

"""
Download the data, if the file isn't already downloaded.
"""
dataset_fname = dataset_url.split("/")[-1]
if not os.path.exists(dataset_fname):
    download(dataset_url, server_fname=dataset_fname)

"""
Get all the data out of the zipfile into a list, so we can start processing.
"""
archive = zipfile.ZipFile(dataset_fname, 'r')
raw = archive.open(archive.infolist()[0]).readlines()
# 0 for ham, 1 for spam
labels = [l.split("\t")[0] for l in raw]
data = [l.split("\t")[1].rstrip() for l in raw]

"""
Let's see some examples from the dataset
"""
for l, d in zip(labels, data)[:10]:
    print("%s %s" % (l, d))

"""
Convert text labels to 0, 1
"""
labels[labels == "ham"] = 0
labels[labels == "spam"] = 1

"""
Want to train on 80% of the data, use last 20% for validation
"""
train_boundary = int(.8 * len(data))
train_X = np.array(data[:train_boundary])
train_y = np.array(labels[:train_boundary])
test_X = np.array(data[train_boundary:])
test_y = np.array(labels[train_boundary:])

"""
Using sklearn's pipelines, this becomes easy
"""
text_cleaner = TfidfVectorizer()
classifier = BernoulliNB()
p = make_pipeline(text_cleaner, classifier)
p.fit(train_X, train_y)

"""
Let's see how it is doing on the training and test sets
"""
pred_train_y = p.predict(train_X)
pred_test_y = p.predict(test_X)
print("Training accuracy %f" % accuracy_score(train_y, pred_train_y))
print("Testing accuracy %f" % accuracy_score(test_y, pred_test_y))
print(" ")
print("Test classification report")
print("==========================")
print(classification_report(test_y, pred_test_y))
