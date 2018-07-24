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

	weightp = model.layers[LIndex].get_weights()[0] #[:,:,0,0]
	weight_cc = flipKernel.flipK(weightp) #weight_cc = np.flipud(np.fliplr(weightp))

	KShape = weight_cc.shape
	numInMaps = KShape[2]
	numOutMaps = KShape[3]
	Kw = Kh = KShape[0]

	if(LIndex==0):
		Win = Hin = model.input_shape[2]
	else:
		Win = Hin = model.layers[LIndex].input_shape[2]

	Wout = (Win - Kw)/1 + 1
	Hout = (Hin - Kh)/1 + 1

	NumInNeurons = numInMaps*Win*Hin
	NumOutNeurons = numOutMaps*Wout*Hout
	NumConn = numOutMaps*Kw*Kh

	ConnMatrix = np.zeros((NumInNeurons, NumConn))
	WtMatrix = np.zeros((NumInNeurons, NumConn))

	for i in range(0,numInMaps):
		for j in range(0,Hin):
			for k in range(0,Win):
				Conn = 0
				Neuron_Index = i*Hin*Win + j*Win + k
				for nF in range(0,numOutMaps):
					for w in range(0,Kw):
						for h in range(0,Kh):
							Out_Index = nF*Wout*Hout + (j-w)*Wout + (k-h)
							ConnMatrix[Neuron_Index, Conn] = Out_Index
							if((j-w>=0) and (k-h>=0) and (j-w <Wout) and (k-h <Hout)):
								WtMatrix[Neuron_Index, Conn] = weight_cc[w,h,i,nF]
							else:
								WtMatrix[Neuron_Index, Conn] = 0
							Conn = Conn + 1

	
	return [ConnMatrix, WtMatrix]
