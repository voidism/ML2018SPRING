import numpy as np
import csv
import math
import random
from math import log, floor
import sys

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Reshape, Permute, multiply
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
# from keras_vggface import utils
from keras.engine.topology import get_source_inputs
import warnings
from keras.models import Model
from keras import layers
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img  

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

def load_data(test_data_path='test.csv'):
    r = csv.reader(open(test_data_path))
    l = list(r)[1:]
    X_test = []
    idx = 0
    for row in l:
        flat_array = np.array(row[1].split(' '),dtype=float)
        X_test.append(np.reshape(flat_array,(48,48,1)))
        idx+=1
    X_test = np.array(X_test,dtype=float)

    return X_test

def test(test,filename,model_name):
    ans = []
    model = load_model(model_name)
    model.summary()
    result = np.argmax(model.predict(test),axis=1)
    for idx in range(result.shape[0]):
        ans.append([idx,result[idx]])

    text = open(filename, "w")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["id", "label"])
    for i in range(len(ans)):
        s.writerow(ans[i])
    text.close()

if __name__ == "__main__":
    model_n = 'my_ensemble_model_71301.h5'
    if sys.argv[3] == 'public':
        model_n = 'my_ensemble_model_71301.h5'
    if sys.argv[3] == 'private':
        model_n = 'my_ensemble_model_70911.h5'
    X_test = load_data(sys.argv[1])
    X_test /= 255
    test(X_test,sys.argv[2],model_n)