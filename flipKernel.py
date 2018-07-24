from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
import os
import numpy as np

def flipK(WtKernel):
	
	WtShape = WtKernel.shape
	numF = WtShape[3]
	numInCh = WtShape[2]
	Kw = WtShape[0]
	Kh = WtShape[1]

	BufMatrix = np.zeros((Kw,Kh))

	for i in range(0,numF):
		for j in range(0,numInCh):
			BufMatrix = np.flipud(np.fliplr(WtKernel[:,:,j,i]))
			WtKernel[:,:,j,i] = BufMatrix

	return WtKernel

	
