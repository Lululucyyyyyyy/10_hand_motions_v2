import os
import io
import sys
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
import tensorflow as tf
from data import get_datasets
import random
from tensorflow.keras import layers
import time
from tensorflow.keras import datasets, layers, models, applications

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

X_train, X_dev, X_test, Y_train, Y_dev, Y_test = get_datasets()
print('datasets retrieved')
print('train:', len(X_train))
print('dev:', len(X_dev))
print('test:', len(X_test))

#define model

model = tf.keras.Sequential()
model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(48, 48, 3), data_format='channels_last'))
model.add(layers.MaxPooling2D((2, 2), (2, 2), padding='SAME', data_format='channels_last'))

model.add(layers.Conv2D(16, (2, 2), activation='relu', data_format='channels_last'))
model.add(layers.Conv2D(16, (2, 2), activation='relu', data_format='channels_last'))
model.add(layers.Conv2D(16, (2, 2), activation='relu', data_format='channels_last'))
model.add(layers.MaxPooling2D((2, 2), (2, 2), padding='SAME', data_format='channels_last'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu', input_dim=16*3*3))
model.add(layers.Dropout(rate=0.5))

model.add(layers.Dense(11, activation='softmax', input_dim=64))

#print summary
model.summary()

#adam = tf.keras.optimizers.Adam(learning_rate='0.01') #tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#train :)
print('==========Training==========')
training = model.fit(X_train, Y_train, epochs=20, steps_per_epoch=37)
plt.plot(training.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


#dev
print('==========Validating==========')
dev_loss, dev_acc = model.evaluate(X_dev, Y_dev)

#test
print('==========Testing==========')
test_loss, test_acc = model.evaluate(X_test, Y_test)

model.save('the_h5_model.h5')
print('the end, saved')