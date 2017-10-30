#import tensorflow as tf
#sess = tf.Session()

#from keras import backend as K
#K.set_session(sess)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
#import cv2

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# from https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
#np.set_printoptions(threshold=np.nan)
'''
testImg = cv2.imread("testimg.jpg")
testImg = testImg / 255
print(testImg.shape)
data[0] = testImg[0]
print(data.shape)
'''

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
'''
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot
(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
'''

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

num_pixels = X_train.shape[1]*X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels)

X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


model = Sequential()

model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(num_classes, kernel_initializer='normal', activation='sigmoid'))

model.summary()
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
#sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
#model.predict(y_train[0], batch_size=32, verbose = 0)

