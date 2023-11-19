import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models, optimizers
import matplotlib.pyplot as plt
import scipy.io
from sklearn.model_selection import train_test_split
import arrow
from collections import Counter

K = 1

# Input dataset.
mat = scipy.io.loadmat('../caltech101_silhouettes_28.mat')
inputs = mat.get('X')
s = inputs.shape
images = np.array([np.reshape(inputs[i,:],[28,28]) for i in range(s[0])])
images = tf.reshape(images,shape=[8671,28,28,1])
images = np.array(images)
labels = np.array(mat.get('Y')).T -1

# Split dataset into train and test.
train_pct = 0.1#0.8#0.4#0.1
valid_pct = 0.1#0.1#0.2#0.1
test_pct = 0.8#0.1#0.4#0.8
train_images,test_images,train_labels,test_labels = train_test_split(images, labels, train_size=train_pct+valid_pct, test_size=test_pct)

print()
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

print(np.unique(train_labels))
print(np.unique(test_labels))
