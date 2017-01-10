import os
import sys
sys.path.append(os.getcwd())
import time
import numpy as np
import cudaModules
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.compiler as compiler
MIDDLELAYER=70
EPOCHS=1000
GAMMA=0.01
PRINTFREQ=100
GPU=False
t0=time.clock()
class Layer:
	def __init__(self,n,m):
		self.A=np.random.randint(-10000,10000,(n,m))/100000.0
		self.A.astype(np.float64)
		self.out=np.array(n*[0]).astype(np.float64)
		self.delta=np.array(n*[0]).astype(np.float64)
		self.deriv=np.array(n*[0]).astype(np.float64)

		kernel_code=cudaModules.forwardTemplate%{'NCOLS':self.A.shape[1]}
		module=compiler.SourceModule(kernel_code)
		self.forwardKernel=module.get_function("forwardKernel")

		kernel_code=cudaModules.deltaTemplate%{'NCOLS':self.A.shape[1]}
		module=compiler.SourceModule(kernel_code)
		self.deltaKernel=module.get_function("deltaKernel")

		kernel_code=cudaModules.weightTemplate%{
			'NCOLS':self.A.shape[1]}
		module=compiler.SourceModule(kernel_code)
		self.weightKernel=module.get_function("weightKernel")

	def insert(self,x):
		if GPU:
			self.forwardKernel(
				drv.In(self.A),
				drv.In(x),
				drv.Out(self.out),
				drv.Out(self.deriv),
				block=(self.A.shape[0],1,1),
				grid=(1,1))
		else:
			self.out=np.tanh(np.dot(self.A,x))
			self.deriv=1.0-(self.out*self.out)
		return self.out
	def updateDelta0(self,y):
		self.delta=self.deriv*y
		return self.delta
	def updateDelta(self,A,y):
		if GPU:
			self.deltaKernel(
				drv.In(self.A),
				drv.InOut(self.delta),
				drv.In(self.out),
				drv.In(self.deriv),
				block=(self.A.shape[1],1,1),
				grid=(1,1))
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
			self.weightKernel(
				drv.InOut(self.A),
				drv.In(x),
				drv.In(self.delta),
				block=(self.A.shape[0],self.A.shape[1],1),
				grid=(1,1))
		else:
			for i in range(self.A.shape[0]):
				for j in range(self.A.shape[1]):
					self.A[i][j] -= self.delta[i]*x[j]
#NN=[Layer(MIDDLELAYER+1,2+1),Layer(2,MIDDLELAYER+1)]
NN=[Layer(MIDDLELAYER,2),Layer(2,MIDDLELAYER)]
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
for i in range(EPOCHS):
	r=np.random.randint(0,4)
#	theInput=np.append(inp[r],[1])
	theInput=inp[r]
	tmp=NN[0].insert(theInput)
	tmp=NN[1].insert(tmp)
	error=tmp-out[r]
	if (i%PRINTFREQ==0):
		print('error:'),
		print(error)
		print('output:'),
		print(tmp)
		print('target:'),
		print(out[r])
	tmp=NN[1].updateDelta0(error)
	tmp=NN[0].updateDelta(NN[1].A,tmp)
	#print(tmp)
	NN[1].updateWeights(NN[0].out)
	NN[0].updateWeights(theInput)
for r in range(4):
#	theInput=np.append(inp[r],[1])
	theInput=inp[r]
	tmp=NN[0].insert(theInput)
	tmp=NN[1].insert(tmp)
	error=tmp-out[r]
	print('error:'),
	print(error)
	print('output:'),
	print(tmp)
	print('target:'),
	print(out[r])
	tmp=NN[1].updateDelta0(error)
	tmp=NN[0].updateDelta(NN[1].A,tmp)
	#print(tmp)
	NN[1].updateWeights(NN[0].out)
	NN[0].updateWeights(theInput)
tf=time.clock()
print('elapsed time: '+str(tf-t0)+'s')
