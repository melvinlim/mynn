import os
import sys
sys.path.append(os.getcwd())
import time
import numpy as np
import cudaModules
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.compiler as compiler
import math
INPUTDIM=2
OUTPUTDIM=2
MIDDLELAYER=1025
MIDDLELAYER1=500
MIDDLELAYER2=10
LAYERS=2
OUTPUTDIM=[MIDDLELAYER,OUTPUTDIM]
INPUTDIM=[INPUTDIM,MIDDLELAYER]
#OUTPUTDIM=[MIDDLELAYER1,MIDDLELAYER2,OUTPUTDIM]
#INPUTDIM=[INPUTDIM,MIDDLELAYER1,MIDDLELAYER2]
EPOCHS=100
GAMMA=0.01
PRINTFREQ=100
GPU=True
TPB1D=512
TPB2D=32
t0=time.clock()
class Layer:
	def __init__(self,n,m):
		self.A=np.random.randint(-10000,10000,(n,m))/100000.0
		self.A.astype(np.float64)
		self.out=np.array(n*[0]).astype(np.float64)
		self.delta=np.array(n*[0]).astype(np.float64)
		self.deriv=np.array(n*[0]).astype(np.float64)

		kernel_code=cudaModules.forwardTemplate%{
			'NROWS':self.A.shape[0],
			'NCOLS':self.A.shape[1]}
		module=compiler.SourceModule(kernel_code)
		self.forwardKernel=module.get_function("forwardKernel")

		kernel_code=cudaModules.deltaTemplate%{
			'NROWS':self.A.shape[0],
			'NCOLS':self.A.shape[1]}
		module=compiler.SourceModule(kernel_code)
		self.deltaKernel=module.get_function("deltaKernel")

		kernel_code=cudaModules.weightTemplate%{
			'NROWS':self.A.shape[0],
			'NCOLS':self.A.shape[1]}
		module=compiler.SourceModule(kernel_code)
		self.weightKernel=module.get_function("weightKernel")

	def insert(self,x):
		if GPU:
			gridY=int(math.ceil(float(self.A.shape[0])/float(TPB1D)))
			self.forwardKernel(
				drv.In(self.A),
				drv.In(x),
				drv.Out(self.out),
				drv.Out(self.deriv),
				block=(1,TPB1D,1),
				grid=(1,gridY))
		else:
			self.out=np.tanh(np.dot(self.A,x))
			self.deriv=1.0-(self.out*self.out)
		return self.out
	def updateDelta0(self,y):
		self.delta=self.deriv*y
		return self.delta
	def updateDelta(self,A,y):
		if GPU:
			gridX=int(math.ceil(float(self.A.shape[1])/float(TPB2D)))
			self.deltaKernel(
				drv.In(self.A),
				drv.InOut(self.delta),
				drv.In(self.out),
				drv.In(self.deriv),
				block=(TPB1D,1,1),
				grid=(gridX,1))
		else:
			arr=[]
			for j in range(len(self.delta)):
				s=0
				for k in range(len(y)):
					s += A[k][j]*y[k]
				arr.append(s)
			self.delta=self.deriv*s
		return self.delta
	def updateWeights(self,x):
		if GPU:
			gridX=int(math.ceil(float(self.A.shape[1])/float(TPB2D)))
			gridY=int(math.ceil(float(self.A.shape[0])/float(TPB2D)))
			self.weightKernel(
				drv.InOut(self.A),
				drv.In(x),
				drv.In(self.delta),
				#block=(self.A.shape[0],self.A.shape[1],1),
				#grid=(1,1))
				block=(TPB2D,TPB2D,1),
				grid=(gridX,gridY))
		else:
			for i in range(self.A.shape[0]):
				for j in range(self.A.shape[1]):
					self.A[i][j] -= self.delta[i]*x[j]
#NN=[Layer(MIDDLELAYER+1,2+1),Layer(2,MIDDLELAYER+1)]
#NN=[Layer(MIDDLELAYER,2),Layer(2,MIDDLELAYER)]
class Network:
	def __init__(self,n):
		self.layer=[]
		self.n=n
		for i in range(n):
			self.layer.append(Layer(OUTPUTDIM[i],INPUTDIM[i]))
	def train(self,theInput,target):
		tmp=theInput
		for i in range(self.n):
			tmp=self.layer[i].insert(tmp)
		output=tmp
		error=tmp-target
		tmp=self.layer[LAYERS-1].updateDelta0(error)
		i=LAYERS-1
		while (i>0):
			tmp=self.layer[i-1].updateDelta(self.layer[i].A,tmp)
			i -= 1
		#print(tmp)
		i=LAYERS-1
		while (i>0):
			self.layer[i].updateWeights(self.layer[i-1].out)
			i -= 1
		self.layer[0].updateWeights(theInput)
		return [output,error]
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
NN=Network(LAYERS)
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
