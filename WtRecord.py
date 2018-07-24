from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
import os
import numpy as np
import conv_indexer 
import fc_indexer
import pool_indexer
import BiasRecord

model_name = 'axsnn.h5' #saved_models/keras_msnn.h5' #'mlp_p.h5'
model = load_model('axsnn.h5') #saved_models/keras_msnn.h5')
model.summary()

numLayers = len(model.layers)
IndexFile = open("AxSNN_ETHAWeight.txt", 'w')

numAL = 0
neuron_count = np.zeros((1,numLayers))
param_record = np.zeros((1,numLayers))
type_record = np.zeros((1,numLayers))

for layer in range(0,numLayers):
	params = model.layers[layer].get_weights()
	oshape = model.layers[layer].output_shape
	
	if(len(params)>0 or len(oshape)>=4):		#Skipping Flatten Layer
		param_record[0,layer] = 1
		if(len(oshape)==4 and len(params)>0):
			neuron_count[0,layer] = oshape[2]*oshape[2]*oshape[1]    #conv2D layer
			type_record[0,layer] = ord('c')
		elif(len(oshape)==4 and len(params)==0):
			neuron_count[0,layer] = oshape[2]*oshape[2]*oshape[1]    #pooling and input layers
			type_record[0,layer] = ord('p')
		elif(len(oshape)==2 and len(params)>0):
			neuron_count[0,layer] = oshape[0]*oshape[1]	#fc layer
			type_record[0,layer] = ord('f')
		numAL = numAL +1

#print(type_record)
IndexFile.write("{:d}".format(int(numAL)))
for i in range(0, numLayers):
	if(param_record[0,i]):
		IndexFile.write("\n{:d}".format(int(neuron_count[0,i])))

ishape = model.input_shape
#print(ishape)

for i in range(0,ishape[2]*ishape[2]): #ishape[1]
	IndexFile.write("\n1 0 0")

for i in range(1, numLayers):  #0 for models where a separate input layer is not present
	params = model.layers[i].get_weights()
	if(param_record[0,i]):
		[BiasMat] = BiasRecord.bIndex(i, type_record[0,i], model_name)
		for j in range(0,int(neuron_count[0,i])):
			IndexFile.write("\n1 0 {:8.6f}".format(BiasMat[0,j]))


IndexFile.write("\n")

for layer in range(1,numLayers):	#0 for models where a separate input layer is not present	#Skipping Input Layer
	params = model.layers[layer].get_weights()	
	print(" Shape of weights : {}".format(len(params)))
	#if(len(params)!=0):
	#print(param_record)
	if(param_record[0,layer]):
		if(type_record[0,layer]== ord('c')):
			[ConnMatrix, WtMatrix] = conv_indexer.create_index(layer, model_name)
		elif(type_record[0,layer]== ord('p')):
			[ConnMatrix, WtMatrix] = pool_indexer.create_index(layer, model_name)
		elif(type_record[0,layer]== ord('f')):
			[ConnMatrix, WtMatrix] = fc_indexer.create_index(layer, model_name)
		
		print(ConnMatrix.shape)
		NumInNeurons = ConnMatrix.shape[0]
		NumConn = ConnMatrix.shape[1]
	
		for i in range(0,NumInNeurons):
			for j in range(0,NumConn):
				if(WtMatrix[i,j]!=0):
					IndexFile.write("{:d}".format(int(ConnMatrix[i,j])))
					IndexFile.write(" ")
					IndexFile.write("{:8.6f}".format(WtMatrix[i,j]))
					IndexFile.write(" ")
			
			IndexFile.write("\n")

IndexFile.close()
