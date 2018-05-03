import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from sklearn.cluster import KMeans
from output import output
import csv

train_num = 140000
X = np.load('image.npy')
X = X.astype('float32') / 255
X = np.reshape(X, (len(X), -1))

x_train = X[:train_num]
x_val = X[train_num:]

input_img = Input(shape=(784,))
encoded = Dense(600, activation='relu')(input_img)
encoded = Dense(500, activation='relu')(encoded)
encoded = Dense(400, activation='relu')(encoded)

decoded = Dense(500, activation='relu')(encoded)
decoded = Dense(600, activation='relu')(decoded)
decoded = Dense(784, activation='relu')(decoded)

encoder = Model(input=input_img, output=encoded)

adam = Adam(lr=5e-4)

autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer=adam, loss='mse')
autoencoder.summary()

autoencoder.fit(x_train, x_train, epochs=50, batch_size=256,
                shuffle=True, validation_data=(x_val, x_val))
autoencoder.save('aucoder.h5')
encoder.save('encoder.h5')

en_img = encoder.predict(X)
en_img = en_img.reshape(en_img.shape[0], -1)
k = KMeans(n_clusters=2, random_state=100).fit(en_img)

print('result:',k.labels_.sum())

filename='test_case.csv'
r = csv.reader(open(filename))
l = list(r)[1:]
ans = []
for i in l:
    ans.append([i[0], int(k.labels_[int(i[1])] == k.labels_[int(i[2])])])

output(ans,name='auto_ans.csv')
