import os
import sys
sys.path.append(os.getcwd())
import time
import numpy as np
import nn

LAYERDIM=[2,1025,2]
#LAYERDIM=[2,500,10,2]
EPOCHS=100
GAMMA=0.01
PRINTFREQ=100
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

np.set_printoptions(precision=4)

NN=nn.Network(LAYERDIM)
for epoch in range(EPOCHS):
	r=np.random.randint(0,4)
	[output,error]=NN.train(inp[r],out[r])
	if (epoch%PRINTFREQ==0):
		print('error:'),
		print(error)
		print('output:'),
		print(output)
		print('target:'),
		print(out[r])
for r in range(4):
	[output,error]=NN.train(inp[r],out[r])
	print('error:'),
	print(error)
	print('output:'),
	print(output)
	print('target:'),
	print(out[r])
tf=time.clock()
print('elapsed time: '+str(tf-t0)+'s')
