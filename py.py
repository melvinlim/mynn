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

#task='xor'
task='linearxor'
#task='csv'
#task='mnist'
[inp,out]=generate(task)

INPUTS=len(inp[0])
OUTPUTS=len(out[0])

BATCHSIZE=10
#raise Exception
#LAYERDIM=[2,1025,2]
#LAYERDIM=[2,500,10,2]
LAYERDIM=[INPUTS,500,OUTPUTS]
EPOCHS=100
GAMMA=0.005
PRINTFREQ=BATCHSIZE

nExamples=len(inp)

np.set_printoptions(precision=4)

NN=nn.Network(LAYERDIM,GAMMA)
filename=task+'.csv'
try:
	open(filename,'r')
	x=raw_input('found '+filename+'.  load network?  ([y]/n)')
	if(x=='' or x=='y'):
		print('loading '+filename+'...')
		NN=nn.loadNetwork(filename)
except:
	x=raw_input(filename+' not found.  start?  ([y]/n)')
	if(x=='n'):
		exit()
t0=time.clock()
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
print('results:')
for r in range(nExamples):
	output=NN.predict(inp[r])
	error=out[r]-output
	printInfo(error,output,out[r])
tf=time.clock()
print('elapsed time: '+str(tf-t0)+'s')
x=raw_input('save network?  (y/[n])')
if(x=='y'):
	NN.save(filename)
