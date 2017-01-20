import os
import sys
sys.path.append(os.getcwd())
import time
import numpy as np
import nn
from csvWrap import *
from filt import *
import extract
from nn import printInfo
from generate import *

#[inp,out]=generate('xor')
#[inp,out]=generate('csv')
[inp,out]=generate('mnist')

INPUTS=len(inp[0])
#INPUTS=129600
#INPUTS=129600/3
#INPUTS=2
#INPUTS=(28*28)

OUTPUTS=len(out[0])
#OUTPUTS=2
#OUTPUTS=10

BATCHSIZE=1
#raise Exception
#LAYERDIM=[2,1025,2]
#LAYERDIM=[2,500,10,2]
LAYERDIM=[INPUTS,500,OUTPUTS]
EPOCHS=10000
GAMMA=0.005
PRINTFREQ=BATCHSIZE
t0=time.clock()

nExamples=len(inp)

np.set_printoptions(precision=4)

NN=nn.Network(LAYERDIM,GAMMA)
for epoch in range(EPOCHS):
	bInp=[]
	bOut=[]
	if BATCHSIZE>1:
		for i in range(BATCHSIZE):
			r=np.random.randint(0,nExamples)
			bInp.append(inp[r])
			bOut.append(out[r])
		[output,error]=NN.batchTrain(bInp,bOut)
		if (epoch%PRINTFREQ==0):
			print('----------epoch:'+str(epoch))
			for i in range(BATCHSIZE):
				printInfo(error[i],output[i],bOut[i])
	else:
		r=np.random.randint(0,nExamples)
		[output,error]=NN.train(inp[r],out[r])
		if (epoch%PRINTFREQ==0):
			print('----------epoch:'+str(epoch))
			printInfo(error,output,out[r])
#for r in range(nExamples):
#	[output,error]=NN.train(inp[r],out[r])
#	printInfo(error,output,out[r])
tf=time.clock()
print('elapsed time: '+str(tf-t0)+'s')
