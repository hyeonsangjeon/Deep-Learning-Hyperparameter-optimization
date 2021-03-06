import matplotlib.pyplot as plt
import numpy as np
import math
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from collections import Counter
import pandas as pd
from tensorflow.keras.callbacks import Callback
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
import custum_score as j

#katib hyperparameter===================================================================
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.00001,
                    help='Initial learning rate')
parser.add_argument('--epochs', type=int, default=2,
                    help='Number of epochs to run trainer.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout ratio')

parser.add_argument('--optimizer', type=str, default='adam',
                   help='the optimizer type')
args = parser.parse_args()
#=======================================================================================

if args.optimizer == 'sgd':
    opt = optimizers.SGD(lr=args.lr, clipvalue=0.5)
elif args.optimizer == 'adam':
    opt = optimizers.Adam(lr=args.lr, clipvalue=0.5)
elif args.optimizer == 'rmsprop':
    opt = optimizers.RMSprop(lr=args.lr, clipvalue=0.5)
else: 
    opt = optimizers.Adam(lr=args.lr, clipvalue=0.5)

#=======================================================================================
# 
class MetricHistory(Callback):
    def on_epoch_end(self, epoch, logs=None):        
        print("\nEpoch {}".format(epoch + 1))
        print("accuracy={:.4f}".format(logs['accuracy']))
        print("loss={:.4f}".format(logs['loss']))
        print("Validation-accuracy={:.4f}".format(logs['val_accuracy']))
        print("Validation-loss={:.4f}".format(logs['val_loss']))
        print("F1-score={:.4f}".format(logs['f1score']))        
        print("recall={:.4f}".format(logs['recall']))
        print("precision={:.4f}".format(logs['precision']))
metric = MetricHistory()
#=======================================================================================


#data load
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


#convert and ready to data
X_train = X_train.reshape(60000, 28,28,1)
X_test = X_test.reshape(10000, 28,28,1)
X_train = X_train/255.0
X_test = X_test/255.0

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

model = Sequential()
# Block 1
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same',  activation='relu', input_shape=(28,28,1,), name='block1_conv1'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same',  activation='relu',  name='block1_conv2'))
model.add(MaxPool2D(pool_size=(2,2), strides=1, name='block1_pool'))

# Block 2
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same',  activation='relu',  name='block2_conv1'))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same',  activation='relu',  name='block2_conv2'))
model.add(MaxPool2D(pool_size=(2,2), strides=2,name='block2_pool'))

# Block 3
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same',  activation='relu',  name='block3_conv1'))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same',  activation='relu',  name='block3_conv2'))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same',  activation='relu',  name='block3_conv3'))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same',  activation='relu',  name='block3_conv4'))
model.add(MaxPool2D(pool_size=(2,2), strides=2,name='block3_pool'))

# Block 4
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same',  activation='relu',  name='block4_conv1'))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same',  activation='relu',  name='block4_conv2'))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same',  activation='relu',  name='block4_conv3'))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same',  activation='relu',  name='block4_conv4'))
model.add(MaxPool2D(pool_size=(2,2), strides=2,name='block4_pool'))

# Block 5
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same',  activation='relu',  name='block5_conv1'))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same',  activation='relu',  name='block5_conv2'))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same',  activation='relu',  name='block5_conv3'))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same',  activation='relu',  name='block5_conv4'))
model.add(MaxPool2D(pool_size=(2,2), strides=2,name='block5_pool'))


model.add(Flatten())
model.add(Dense(1024, activation='relu', name='fc1'))
model.add(Dropout(args.dropout))   # 
model.add(BatchNormalization())
model.add(Dense(128, activation='relu', name='fc2'))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax', name='predictions')) #

          
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', j.f1score, j.recall, j.precision])

hist = model.fit(X_train, Y_train, batch_size=200, epochs=args.epochs, validation_split=0.2, callbacks=[metric])

