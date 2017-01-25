import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import cudaModules
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.compiler as compiler
from pycuda.autoinit import context
import math
from copy import *
TESTGPU=False
TOL=0.01
GPU=False
TPB1D=512
TPB2D=32
class cudaKernels:
	def __init__(self,m,n,mNext,nNext,gamma,ADAGAMMA,EPSILON,adaDelta):

		kernel_code=cudaModules.forwardTemplate%{
			'NROWS':m,
			'NCOLS':n}
		module=compiler.SourceModule(kernel_code)
		self.forwardKernel=module.get_function("forwardKernel")

		if nNext>0:
			kernel_code=cudaModules.deltaTemplate%{
				'NROWS':mNext,
				'NCOLS':nNext}
			module=compiler.SourceModule(kernel_code)
			self.deltaKernel=module.get_function("deltaKernel")

		kernel_code=cudaModules.batchAccumTemplate%{
			'GAMMA':gamma,
			'NROWS':m,
			'NCOLS':n}
		module=compiler.SourceModule(kernel_code)
		self.batchAccumKernel=module.get_function("batchAccumKernel")

		if adaDelta==True:
			kernel_code=cudaModules.batchUpdateADTemplate%{
				'GAMMA':ADAGAMMA,
				'EPSILON':EPSILON,
				'NROWS':m,
				'NCOLS':n}
			module=compiler.SourceModule(kernel_code)
			self.batchUpdateKernel=module.get_function("batchUpdateADKernel")
		else:
			kernel_code=cudaModules.batchUpdateTemplate%{
				'GAMMA':gamma,
				'NROWS':m,
				'NCOLS':n}
			module=compiler.SourceModule(kernel_code)
			self.batchUpdateKernel=module.get_function("batchUpdateKernel")

		kernel_code=cudaModules.weightTemplate%{
			'GAMMA':gamma,
			'NROWS':m,
			'NCOLS':n}
		module=compiler.SourceModule(kernel_code)
		self.weightKernel=module.get_function("weightKernel")
class Layer:
	def __init__(self,m,n,mNext,nNext,gamma,ADAGAMMA=0.95,EPSILON=0.000001,adaDelta=True):
		self.gamma=gamma

		self.A=np.random.randint(-10000,10000,(m,n))/100000.0
		self.A.astype(np.float64)
		self.dA=np.zeros_like(self.A)
		self.out=np.array(m*[0]).astype(np.float64)
		self.delta=np.zeros_like(self.out)
		self.deriv=np.zeros_like(self.out)
		self.mNext=mNext
		self.nNext=nNext

		self.ADAGAMMA=ADAGAMMA
		self.EPSILON=EPSILON
		self.adaDelta=adaDelta

		if adaDelta==True:
			self.grad2=np.zeros_like(self.A)
			self.theta2=np.zeros_like(self.A)

		self.initKernels()
#		self.kernels=cudaKernels(m,n,mNext,nNext,gamma,ADAGAMMA,EPSILON,adaDelta)

	def initKernels(self):
		self.kernels=cudaKernels(self.A.shape[0],self.A.shape[1],self.mNext,self.nNext,self.gamma,self.ADAGAMMA,self.EPSILON,self.adaDelta)

	def hAlloc(self,x):
		ret=drv.mem_alloc(x.nbytes)
		drv.memcpy_htod(ret,x)
		return ret

	def forwardGPU(self,x):
		gA=self.hAlloc(self.A)
		gx=self.hAlloc(x)
		gout=self.hAlloc(self.out)
		gderiv=self.hAlloc(self.deriv)
		gridY=int(math.ceil(float(self.A.shape[0])/float(TPB1D)))
		self.kernels.forwardKernel(
			gA,gx,gout,gderiv,
			block=(1,TPB1D,1),
			grid=(1,gridY))
		t1=np.zeros_like(self.out)
		t2=np.zeros_like(self.deriv)
		drv.memcpy_dtoh(t1,gout)
		drv.memcpy_dtoh(t2,gderiv)
		return [t1,t2]

	def forwardCPU(self,x):
		out=np.tanh(np.dot(self.A,x))
		deriv=1.0-(out*out)
		return [out,deriv]

	def insert(self,x):
		context.synchronize()
		if TESTGPU:
			[t1,t2]=self.forwardGPU(x)
			[self.out,self.deriv]=self.forwardCPU(x)
			for i in range(len(self.out)):
				assert np.fabs(t1[i]-self.out[i])<TOL
				assert np.fabs(t2[i]-self.deriv[i])<TOL
		elif GPU:
			[self.out,self.deriv]=self.forwardGPU(x)
		else:
			[self.out,self.deriv]=self.forwardCPU(x)
		context.synchronize()
		return self.out
	def updateDelta0(self,error):
		self.delta=self.deriv*error
		return self.delta
	def updateDeltaGPU(self,A,y):
		gridX=int(math.ceil(float(A.shape[0])/float(TPB1D)))

		kernel_code=cudaModules.deltaTemplate%{
			'NROWS':A.shape[0],
			'NCOLS':A.shape[1]}
		module=compiler.SourceModule(kernel_code)
		self.kernels.deltaKernel=module.get_function("deltaKernel")

		t1=np.zeros_like(self.delta)
		self.kernels.deltaKernel(
			drv.In(A),
			drv.Out(t1),
			drv.In(y),
			drv.In(self.deriv),
			block=(TPB1D,1,1),
			grid=(gridX,1))
		return t1
	def updateDeltaCPU(self,A,y):
		delta=np.zeros_like(self.delta)
		for j in range(len(delta)):
			s=np.array(0).astype(np.float64)
			for k in range(len(y)):
				s += A[k][j]*y[k]
			delta[j]=self.deriv[j]*s
		return delta
	#delta=dE/dx
	def updateDelta(self,A,y):
		assert(y.size==A.shape[0])
		assert(self.deriv.size==A.shape[1])
		assert(self.delta.size==A.shape[1])
		if TESTGPU:
			t1=self.updateDeltaGPU(A,y)
			self.delta=self.updateDeltaCPU(A,y)
			for i in range(len(t1)):
				assert np.fabs(self.delta[i]-t1[i])<TOL
			self.delta=t1
		elif GPU:
			self.delta=self.updateDeltaGPU(A,y)
		else:
			self.delta=self.updateDeltaCPU(A,y)
		return self.delta
	def batchAccumGPU(self,x):
		gdA=self.hAlloc(self.dA)
		gx=self.hAlloc(x)
		gdelta=self.hAlloc(self.delta)
		gridX=int(math.ceil(float(self.dA.shape[1])/float(TPB2D)))
		gridY=int(math.ceil(float(self.dA.shape[0])/float(TPB2D)))
		self.kernels.batchAccumKernel(
			gdA,
			gx,
			gdelta,
			block=(TPB2D,TPB2D,1),
			grid=(gridX,gridY))
		t1=np.zeros_like(self.dA)
		drv.memcpy_dtoh(t1,gdA)
		return t1
	def batchAccumCPU(self,x):
		t1=np.zeros_like(self.dA)
		for i in range(self.dA.shape[0]):
			for j in range(self.dA.shape[1]):
				t1[i][j]=self.dA[i][j]+self.delta[i]*x[j]
		return t1
	def batchAccum(self,x):
		if TESTGPU:
			t1=self.batchAccumGPU(x)
			t2=self.batchAccumCPU(x)
			for i in range(self.dA.shape[0]):
				for j in range(self.dA.shape[1]):
					assert np.fabs(t1[i][j]-t2[i][j])<TOL
			self.dA=t1
		elif GPU:
			self.dA=self.batchAccumGPU(x)
		else:
			for i in range(self.dA.shape[0]):
				for j in range(self.dA.shape[1]):
					self.dA[i][j] += self.delta[i]*x[j]
	def batchUpdateGPU(self):
		ret=[]
		ans=np.zeros_like(self.A)
		g2=np.zeros_like(self.grad2)
		t2=np.zeros_like(self.theta2)
		if self.adaDelta:
			gA=self.hAlloc(self.A)
			gdA=self.hAlloc(self.dA)
			ggrad2=self.hAlloc(self.grad2)
			gtheta2=self.hAlloc(self.theta2)
			gridX=int(math.ceil(float(self.dA.shape[1])/float(TPB2D)))
			gridY=int(math.ceil(float(self.dA.shape[0])/float(TPB2D)))
			self.kernels.batchUpdateKernel(
				gA,
				gdA,
				ggrad2,
				gtheta2,
				block=(TPB2D,TPB2D,1),
				grid=(gridX,gridY))
			drv.memcpy_dtoh(ans,gA)
			drv.memcpy_dtoh(g2,ggrad2)
			drv.memcpy_dtoh(t2,gtheta2)
			ret.append(ans)
			ret.append(g2)
			ret.append(t2)
		else:
			gA=self.hAlloc(self.A)
			gdA=self.hAlloc(self.dA)
			gridX=int(math.ceil(float(self.dA.shape[1])/float(TPB2D)))
			gridY=int(math.ceil(float(self.dA.shape[0])/float(TPB2D)))
			self.kernels.batchUpdateKernel(
				gA,
				gdA,
				block=(TPB2D,TPB2D,1),
				grid=(gridX,gridY))
			drv.memcpy_dtoh(ans,gA)
			ret.append(ans)
		return ret
	def batchUpdateCPU(self):
		ans=np.zeros_like(self.A)
		if self.adaDelta:
			for i in range(self.A.shape[0]):
				for j in range(self.A.shape[1]):
					self.grad2[i][j]=self.ADAGAMMA*self.grad2[i][j]+(1-self.ADAGAMMA)*(self.dA[i][j]**2)
					theta=(-1)*np.sqrt(self.theta2[i][j]+self.EPSILON)/(np.sqrt(self.grad2[i][j]+self.EPSILON))*self.dA[i][j]
					self.theta2[i][j]=self.ADAGAMMA*self.theta2[i][j]+(1-self.ADAGAMMA)*(theta**2)
					ans[i][j] = self.A[i][j] + theta
		else:
			for i in range(self.A.shape[0]):
				for j in range(self.A.shape[1]):
					ans[i][j] = self.A[i][j] - self.gamma*self.dA[i][j]
		return ans
	def batchUpdate(self):
		if TESTGPU:
			[t1,x,y]=self.batchUpdateGPU()
			t2=self.batchUpdateCPU()
			for i in range(self.dA.shape[0]):
				for j in range(self.dA.shape[1]):
					assert np.fabs(t1[i][j]-t2[i][j])<TOL
			self.A=t1
		elif GPU:
			[self.A,x,y]=self.batchUpdateGPU()
		else:
			self.A=self.batchUpdateCPU()
	def batchInit(self):
		self.dA=np.zeros_like(self.A)
	def updateWeights(self,x):
		if GPU:
			gridX=int(math.ceil(float(self.A.shape[1])/float(TPB2D)))
			gridY=int(math.ceil(float(self.A.shape[0])/float(TPB2D)))
			self.kernels.weightKernel(
				drv.InOut(self.A),
				drv.In(x),
				drv.In(self.delta),
				block=(TPB2D,TPB2D,1),
				grid=(gridX,gridY))
		else:
			for i in range(self.A.shape[0]):
				for j in range(self.A.shape[1]):
					self.A[i][j] -= self.gamma*self.delta[i]*x[j]
class Network:
	def __init__(self,layerdims,gamma):
		self.layer=[]
		ninputs=layerdims[0]
		noutputs=layerdims[-1]
		self.n=len(layerdims)-1
		outputdims=[]
		inputdims=[ninputs]
		for i in layerdims[1:-1]:
			outputdims.append(i)
			inputdims.append(i)
		outputdims.append(noutputs)
		for i in range(self.n-1):
			self.layer.append(Layer(outputdims[i],inputdims[i],outputdims[i+1],inputdims[i+1],gamma))
		#self.layer.append(Layer(outputdims[self.n-1],inputdims[self.n-1],0,0,gamma))
		self.layer.append(LinearLayer(outputdims[self.n-1],inputdims[self.n-1],0,0,gamma))
	def predict(self,theInput):
		tmp=theInput
		for i in range(self.n):
			tmp=self.layer[i].insert(tmp)
		output=deepcopy(tmp)
		return output
	def train(self,theInput,target):
		tmp=theInput
		for i in range(self.n):
			tmp=self.layer[i].insert(tmp)
		output=deepcopy(tmp)
		error=tmp-target
		tmp=self.layer[self.n-1].updateDelta0(error)
		i=self.n-1
		while (i>0):
			tmp=self.layer[i-1].updateDelta(self.layer[i].A,tmp)
			i -= 1
		#print(tmp)
		i=self.n-1
		while (i>0):
			self.layer[i].updateWeights(self.layer[i-1].out)
			i -= 1
		self.layer[0].updateWeights(theInput)
		return [output,error]
	def batchUpdate(self,theInput,target):
		tmp=theInput
		for i in range(self.n):
			tmp=self.layer[i].insert(tmp)
		output=tmp
		error=tmp-target
		tmp=self.layer[self.n-1].updateDelta0(error)
		i=self.n-1
		while (i>0):
			tmp=self.layer[i-1].updateDelta(self.layer[i].A,tmp)
			i -= 1
		i=self.n-1
		while (i>0):
			self.layer[i].batchAccum(self.layer[i-1].out)
			i -= 1
		self.layer[0].batchAccum(theInput)
		return [output,error]
	def batchTrain(self,inputList,targetList):
		for i in range(self.n):
			self.layer[i].batchInit()
		batchsize=len(inputList)
		error=[]
		output=[]
		for i in range(batchsize):
			[o,e]=self.batchUpdate(inputList[i],targetList[i])
			error.append(e)
			output.append(deepcopy(o))
		for i in range(self.n):
			self.layer[i].batchUpdate()
		return [output,error]
	def save(self,filename):
		import pickle
		for i in range(self.n):
			self.layer[i].kernels=0
		fp=open(filename,'w')
		pickle.dump(self,fp)
class LinearLayer(Layer):
	def forwardGPU(self,x):
		raise NotImplementedError('unimplemented.')
	def forwardCPU(self,x):
		out=np.dot(self.A,x)
		deriv=1.0
		return [out,deriv]
def loadNetwork(filename):
	import pickle
	try:
		fp=open(filename,'r')
	except:
		print('failed to open '+filename)
	net=pickle.load(fp)
	for i in range(net.n):
		net.layer[i].initKernels()
	return net
def printInfo(error,output,target):
	print('error:'),
	print(error)
	print('output:'),
	print(output)
	print('target:'),
	print(target)
	print('-')
