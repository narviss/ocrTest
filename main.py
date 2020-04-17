from mnist import MNIST
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
from sklearn.metrics import accuracy_score
import numpy as np
import os
import scipy.misc
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

print("Loading dataset...")
mndata = MNIST("./data/")
images, labels = mndata.load_training()

clf = KNeighborsClassifier()

train_x = images[:10000]
train_y = labels[:10000]

print("Train model")
clf.fit(train_x, train_y)

# Test on the next 100 images:
test_x = images[10000:10100]
expected = labels[10000:10100].tolist()

print("Compute predictions")
predicted = clf.predict(test_x)

print("Accuracy: ", accuracy_score(expected, predicted))

# Test on my Image
image_file_name = os.path.join('output.png')
if True:
    imgFull = Image.open(image_file_name).convert("L")
    count = 0
    imgFull = np.resize(imgFull, (28, 28, 1))
    imgFull = np.array(imgFull)
    im2arr = imgFull.reshape(1, -1).tolist()
    predicted = clf.predict(im2arr)
    print(predicted)
