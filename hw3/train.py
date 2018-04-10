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
from counter import counter
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

def augment_img(X, Y, datagen, expand_size):
    result_x = []
    result_y = []
    batch_size=32
    total_img = len(X)//batch_size+int(len(X)%batch_size!=0)
    ct = counter(epoch=total_img,title="Data Augmentation")
    for i in range(total_img):
        x = X[batch_size*i:batch_size*(i+1)]
        y = Y[batch_size*i:batch_size*(i+1)]
        iter = 0
        for batch_x, batch_y in datagen.flow(x,y, batch_size=32):  
            # print("batch_x",batch_x.shape)
            # print("batch_y",batch_y.shape)
            result_x = result_x + list(batch_x)
            result_y = result_y + list(batch_y)
            iter+=1
            if iter >= expand_size:  
                break  # otherwise the generator would loop indefinitely 
        ct.flush(i)
    result_x = np.array(result_x)
    result_y = np.array(result_y)
    # result_x = np.reshape(result_x,result_x.shape[:3])
    return result_x, result_y

def augmentation(X_train, Y_train, expand_size=5):
    datagen = ImageDataGenerator(  
        rotation_range=0.2,  
        width_shift_range=0,  
        height_shift_range=0,  
        shear_range=0.2,  
        zoom_range=0.2,  
        horizontal_flip=True,  
        fill_mode='nearest')

    X_aug, Y_aug  = augment_img(X_train,Y_train,datagen, expand_size)
    X_train = np.concatenate((X_train, X_aug),axis=0)
    Y_train = np.concatenate((Y_train, Y_aug),axis=0)

    randomize = np.arange(len(X_train))
    np.random.shuffle(randomize)
    X_train,Y_train = (X_train[randomize], Y_train[randomize])

    return X_train, Y_train

def load_data(train_data_path='train.csv',test_data_path='test.csv'):
    r = csv.reader(open(train_data_path))
    l = list(r)[1:]
    X_train = []
    Y_pre = []
    idx = 0
    ct = counter(epoch=len(l),title="Loading Training Data")
    for row in l:
        Y_pre.append(row[0])
        flat_array = np.array(row[1].split(' '),dtype=float)
        # mu = np.mean(flat_array)
        # sigma = np.std(flat_array)
        # flat_array = (flat_array - mu) / sigma
        X_train.append(np.reshape(flat_array,(48,48,1)))
        ct.flush(j=idx)
        idx+=1
    X_train = np.array(X_train,dtype=float)

    # one-hot encoding
    Y_pre = np.array(Y_pre,dtype=int)
    Y_train = np.zeros((Y_pre.shape[0],7))
    Y_train[np.arange(Y_pre.shape[0]), Y_pre] = 1

    r = csv.reader(open(test_data_path))
    l = list(r)[1:]
    X_test = []
    idx = 0
    ct = counter(epoch=len(l),title="Loading Testing Data")
    for row in l:
        flat_array = np.array(row[1].split(' '),dtype=float)
        # mu = np.mean(flat_array)
        # sigma = np.std(flat_array)
        # flat_array = (flat_array - mu) / sigma
        X_test.append(np.reshape(flat_array,(48,48,1)))
        ct.flush(idx)
        idx+=1
    X_test = np.array(X_test,dtype=float)

    return X_train, Y_train, X_test

def normalization(X_train):
    X_train /= 255
    return X_train

def split_valid(X,Y,v_size=0.9,rand=False,split=0,block=0):
    if rand:
        randomize = np.arange(len(X))
        np.random.shuffle(randomize)
        X,Y = (X[randomize], Y[randomize])
    
    v_size = floor(len(X) * v_size)
    Vx = X[v_size:] if v_size != len(X) else X[floor(len(X) * 0.9):]
    Vy = Y[v_size:] if v_size != len(X) else Y[floor(len(X) * 0.9):]
    X = X[:v_size]
    Y = Y[:v_size]

    if split!=0:
        v_size = floor(len(X) // split)
        block-=1
        if block not in range(split):
            block = 0
            print("please choose the right block!")
        Vx = X[block*v_size:(block+1)*v_size]
        Vy = Y[block*v_size:(block+1)*v_size]
        X = np.concatenate((X[:block*v_size],X[(block+1)*v_size:]))
        Y = np.concatenate((Y[:block*v_size],Y[(block+1)*v_size:]))
    
    return X, Y, Vx, Vy


def build_model(x_train):
    num_classes = 7

    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3)))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3)))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.333))
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.333))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))


    return model

def train_model(model,x_train,y_train,x_test,y_test,gen=False):

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    ''' original:
    model.fit(x_train, y_train,
              batch_size=300,
              epochs=40,
              validation_data=(x_test, y_test),
              shuffle=True)
    '''
    # model.fit_generator(datagen.flow(x_train, y_train, batch_size=50),
    #                 steps_per_epoch=len(x_train) / 32, epochs=50,
    #           validation_data=(x_test, y_test))
    # model.fit_generator(datagen.flow(x_train, y_train, batch_size=100,
    #           shuffle=True),
    #           epochs=50,
    #           steps_per_epoch=len(x_train) / 1000,
    #           validation_data=(x_test, y_test))

    LOG_DIR = './training_logs'
    LOG_FILE_PATH = LOG_DIR + '/checkpoint-{epoch:02d}-{val_loss:.4f}.hdf5'   # 模型Log文件以及.h5模型文件存放地址

    tensorboard = TensorBoard(log_dir=LOG_DIR, write_images=True)
    checkpoint = ModelCheckpoint(filepath=LOG_FILE_PATH, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1)

    if gen:
        datagen = ImageDataGenerator(  
            rotation_range=0.2,  
            width_shift_range=0.2,  
            height_shift_range=0.2,  
            shear_range=0.2,  
            zoom_range=0.2,  
            horizontal_flip=True,  
            fill_mode='nearest')

        datagen.fit(x_train)
        model.fit_generator(datagen.flow(x_train,  
                            y_train, batch_size=100),
                        steps_per_epoch=round(len(x_train)/64),
                        epochs=40, validation_data=(x_test, y_test),
                        callbacks=[tensorboard, checkpoint, early_stopping])
    else:
        model.fit(x_train, y_train,
                batch_size=256,
                epochs=40,
                validation_data=(x_test, y_test),
                shuffle=True,
                callbacks=[tensorboard, checkpoint, early_stopping])

    return model

def test(test,filename = "ans.csv",model_name='my_model.h5',checkname=""):
    ans = []
    model = None
    if checkname=="":
        model = load_model('my_model.h5')
    else:
        model = load_model('training_logs/checkpoint-'+checkname+'.hdf5')
    result = np.argmax(model.predict(test),axis=1)
    for idx in range(result.shape[0]):
        ans.append([idx,result[idx]])

    text = open(filename, "w")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["id", "label"])
    for i in range(len(ans)):
        s.writerow(ans[i])
    text.close()
    
if __name__=="__main__":
    X_train, Y_train, X_test = load_data()
    X_train, Y_train, X_valid, Y_valid = split_valid(X_train, Y_train)

    if "-aug" in sys.argv:
        X_train, Y_train = augmentation(X_train, Y_train)
    
    X_train = normalization(X_train)
    X_test = normalization(X_test)
    X_valid = normalization(X_valid)

    if '-init' in sys.argv:
        model = build_model(X_train)
    elif '-good' in sys.argv:
        model = load_model('my_model_init.h5')
    elif '-cont' in sys.argv:
        model = load_model('my_model.h5')
    else:
        print('No Arguments!')
        sys.exit()
    model = train_model(model, X_train, Y_train, X_valid, Y_valid)
    model.save('my_model.h5')
    test(X_test)
