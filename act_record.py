from __future__ import absolute_import
from __future__ import print_function
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, AveragePooling2D
from keras.layers import MaxPooling2D, BatchNormalization
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import os
import numpy as np

model = load_model('models/keras_mnist.12-0.99.h5')
model.summary()
batch_size = 128
nb_classes = 10
nb_epoch = 12
img_rows, img_cols = 28, 28 # input image dimensions

# number of convolutional filters to use
filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
# color channels
chnls = 1


# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], chnls, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], chnls, img_rows, img_cols)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

str1 = "mnist99_act_keras"
str2 = 1
str4 = "AxSNN_KInput_ConvNet"
str3 = ".txt"

num_Layers = len(model.layers)
print("Num Layers : {}".format(num_Layers))

#print("\n {}".format(weights[1][0][0][0]))
weightm = model.layers[0].get_weights()
weightp = model.layers[0].get_weights()[0][:,:,0,0]
#weightm[1] = model.layers[0].get_weights()[1]

#for i in range(0,32):
#	weightm[1][i] = 0

print(X_test.shape)
print(weightp.shape)
weightp = np.expand_dims(weightp, axis =2)
print(weightp.shape)
weightp = np.expand_dims(weightp, axis =3)
print(weightp.shape)
model2 = Sequential()

inp = np.ones((1,28,28))
inp = np.expand_dims(inp, axis = 1)


model2.add(Conv2D(filters, nb_conv, nb_conv,input_shape=(1, img_rows, img_cols),weights = model.layers[0].get_weights())) #[weightp,weightm[1][0:1]]))
#model.add(Activation('relu'))
activations = model2.predict(X_test)
model2.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=['accuracy'])
model2.save('kmodel_all.h5')

weightd = [weightp, weightm[1][0:1]]
print(weightd)

W = 26
H = 26
NumChannels = 32

print(activations.shape)
ActFile = open(str1+str3, 'w')
for ch in range(0,32):
	for i in range(0,26):
		for j in range(0,26):
			if(activations[0][ch][i][j]>0):
			#	ActFile.write("\n{:8.6f}".format(i))
			#	ActFile.write(" {:8.6f}".format(j))
				ActFile.write(" {:8.6f}".format(activations[0][ch][i][j]))
			
		

#print(activations)
