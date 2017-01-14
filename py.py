import os
import sys
sys.path.append(os.getcwd())
import time
import numpy as np
import nn
from csvWrap import *
from filt import *

INPUTS=129600
INPUTS=129600/3
#INPUTS=2

BATCHSIZE=1

#LAYERDIM=[2,1025,2]
#LAYERDIM=[2,500,10,2]
LAYERDIM=[INPUTS,40,2]
EPOCHS=1000
GAMMA=0.0001
PRINTFREQ=10
GPU=True
t0=time.clock()

inp1=np.array([-1,-1]).astype(np.float64)
inp2=np.array([-1,+1]).astype(np.float64)
inp3=np.array([+1,-1]).astype(np.float64)
inp4=np.array([+1,+1]).astype(np.float64)
inp=[inp1,inp2,inp3,inp4]

out1=np.array([-1,-1]).astype(np.float64)
out2=np.array([-1,+1]).astype(np.float64)
out3=np.array([+1,-1]).astype(np.float64)
out4=np.array([+1,+1]).astype(np.float64)
out=[out1,out2,out3,out4]

NPOS=24
NNEG=24

inp=[]
out=[]

#filt=Filter(1,INPUTS,1,0,255,False)
filt=Filter(1,INPUTS,1,0,255*3,False)

for i in range(NPOS):
	filename='verified/pos'+str(i)+'.csv'
	x=readCSV(filename)
	t=np.array(map(int,x)).astype(np.float64)
	t=t.reshape(-1,len(t)/3,3).sum(axis=2)[0]
	y=filt.insert1D(t)
	inp.append(y)
	out.append([-1,+1])

for i in range(NNEG):
	filename='verified/neg'+str(i)+'.csv'
	x=readCSV(filename)
	t=np.array(map(int,x)).astype(np.float64)
	t=t.reshape(-1,len(t)/3,3).sum(axis=2)[0]
	y=filt.insert1D(t)
	inp.append(y)
	out.append([+1,-1])

#raise Exception
#inp=[inp1,inp2,inp3,inp4]
#out=[out1,out2,out3,out4]
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
			for i in range(BATCHSIZE):
				print('error:'),
				print(error[i])
				print('output:'),
				print(output[i])
				print('target:'),
				print(bOut[i])
	else:
		r=np.random.randint(0,nExamples)
		[output,error]=NN.train(inp[r],out[r])
		if (epoch%PRINTFREQ==0):
			print('error:'),
			print(error)
			print('output:'),
			print(output)
			print('target:'),
			print(out[r])
for r in range(nExamples):
	[output,error]=NN.train(inp[r],out[r])
	print('error:'),
	print(error)
	print('output:'),
	print(output)
	print('target:'),
	print(out[r])
tf=time.clock()
print('elapsed time: '+str(tf-t0)+'s')
