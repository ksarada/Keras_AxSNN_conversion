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

	KShape = model.layers[LIndex].output_shape
	numInMaps = numOutMaps = KShape[1]
	Kw = Kh = np.floor(model.layers[LIndex].input_shape[2]/model.layers[LIndex].output_shape[2]) #2 #KShape[0]
	Win = Hin = model.layers[LIndex].input_shape[2]
	Ksize = Kw 
	Wout = np.floor((Win - Kw)/Ksize) + 1
	Hout = np.floor((Hin - Kh)/Ksize) + 1

	NumInNeurons = numInMaps*Win*Hin
	NumOutNeurons = numOutMaps*Wout*Hout
	NumConn = 1

	ConnMatrix = np.zeros((NumInNeurons, NumConn))
	WtMatrix = np.zeros((NumInNeurons, NumConn))

	for i in range(0,numInMaps):
		for j in range(0,Hin):
			for k in range(0,Win):
				Neuron_Index = i*Hin*Win + j*Win + k
				Out_Index = i*Wout*Hout + (np.floor(j/2))*Wout + np.floor(k/2)
				ConnMatrix[Neuron_Index,0] = Out_Index
				WtMatrix[Neuron_Index,0] = 1/(Ksize*Ksize) 

	
	return [ConnMatrix, WtMatrix]
