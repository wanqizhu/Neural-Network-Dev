import numpy as np
from urllib.request import urlretrieve
import gzip
import os


def get_file(filename, source, offset=16):
    download(filename, source)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=offset)
    return data


def download(filename, source):
    if not os.path.exists(filename):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)


def vec_to_onehot(vec):
    num_categories = len(np.unique(vec)) # number of unique values
    vec_placeholder = np.zeros((vec.shape[0], num_categories),
                               dtype='float32') #initialize array of dimension (vec.shape[0], num_categories)
    vec_placeholder[np.arange(vec.shape[0]), vec] = 1
    return vec_placeholder


def get_val_set(x_train, y_train, val_size):
    x_train, x_val = x_train[:-val_size], x_train[-val_size:]
    y_train, y_val = y_train[:-val_size], y_train[-val_size:]
    return x_train, y_train, x_val, y_val


def load_mnist(val_size=10000):
    print("Loading data...")
    source = 'http://yann.lecun.com/exdb/mnist/'

    def load_mnist_images(filename):
        data = get_file(filename, source)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        return get_file(filename, source, offset=8)

    x_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    x_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    y_train, y_test = vec_to_onehot(y_train), vec_to_onehot(y_test)
    x_train, y_train, x_val, y_val = get_val_set(x_train, y_train, val_size)

    return x_train, y_train, x_val, y_val, x_test, y_test
