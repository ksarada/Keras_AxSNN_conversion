from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
import os
import numpy as np
import flipKernel

def create_index(LIndex, Model):

	model = load_model(Model) #'models/keras_mnist.12-0.99.h5')
	model.summary()

	weightp = model.layers[LIndex].get_weights()[0] 

	KShape = weightp.shape
	NumInNeurons = KShape[0]
	NumOutNeurons = KShape[1]
	NumConn = NumOutNeurons

	ConnMatrix = np.zeros((NumInNeurons, NumConn))
	WtMatrix = np.zeros((NumInNeurons, NumConn))

	for i in range(0,NumInNeurons):
		for j in range(0,NumOutNeurons):
				Conn = j
				Neuron_Index = i 
				Out_Index = j
				ConnMatrix[Neuron_Index, Conn] = Out_Index
				WtMatrix[Neuron_Index, Conn] = weightp[i,j]

	
	return [ConnMatrix, WtMatrix]
