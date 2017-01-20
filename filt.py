import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.compiler as compiler
import math
TPB1D=512
TPB2D=32
normalize1DTemplate="""
__global__ void normalize1dKernel(double *input,double *output){
	const int row = blockIdx.y * blockDim.y + threadIdx.y;

	const double scale=%(MAX)s-%(MIN)s;
	const int shift=scale/2.0;

	if((row<%(YDIM)s)){
		output[row]=(input[row]-%(MIN)s-shift)/scale*2.0;
	}
}
"""
normalize2DTemplate="""
__global__ void normalize2DKernel(double *input,double *output){
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int row = blockIdx.y * blockDim.y + threadIdx.y;

	const double scale=%(MAX)s-%(MIN)s;
	const int shift=scale/2.0;

	if((col<%(XDIM)s)&&(row<%(YDIM)s)){
		output[row*%(XDIM)s+col]=(input[row*%(XDIM)s+col]-%(MIN)s-shift)/scale*2.0;
	}
}
"""
class Filter:
	def __init__(self,x=1,y=1,z=1,a=0,b=255,GPU=True):
		self.output=np.array(x*y*z*[0]).astype(np.float64)
		if x==1:
			self.output.reshape(y)
		elif z==1:
			self.output.reshape(x,y)
		else:
			self.output.reshape(x,y,z)
		self.x=x
		self.y=y
		self.z=z
		self.a=a
		self.b=b
		self.GPU=GPU

		if GPU:
			kernel_code=normalize1DTemplate%{
				'YDIM':self.y,
				'MIN':self.a,
				'MAX':self.b
			}
			module=compiler.SourceModule(kernel_code)
			self.normalize1DKernel=module.get_function("normalize1DKernel")
			kernel_code=normalize2DTemplate%{
				'XDIM':self.x,
				'YDIM':self.y,
				'MIN':self.a,
				'MAX':self.b
			}
			module=compiler.SourceModule(kernel_code)
			self.normalize2DKernel=module.get_function("normalize2DKernel")

	def insert1D(self,inp):
		if self.GPU:
			gridY=int(math.ceil(float(self.y)/float(TPB1D)))
			self.normalize1DKernel(
				drv.In(inp),
				drv.Out(self.output),
				block=(1,TPB1D,1),
				grid=(1,gridY))
		else:
			scale=float(self.b-self.a)
			shift=int(scale/2)
			self.output=(inp-self.a-shift)/(scale)*2.0
		return self.output

	def insert2D(self,inp):
		if self.GPU:
			gridX=int(math.ceil(float(self.x)/float(TPB2D)))
			gridY=int(math.ceil(float(self.y)/float(TPB2D)))
			self.normalize2DKernel(
				drv.In(inp),
				drv.Out(self.output),
				block=(TPB2D,TPB2D,1),
				grid=(gridX,gridY))
		else:
			scale=float(self.b-self.a)
			shift=int(scale/2)
			self.output=(inp-self.a-shift)/(scale)*2.0
		return self.output

import time
def test1D(y,a,b,GPU):
	testFilt=Filter(1,y,1,a,b,False)
	x=np.random.randint(a,b,(y))
	t0=time.clock()
	y=testFilt.insert1D(x)
	t1=time.clock()
	print(x)
	print(y)
	print('min ='+str(np.min(y)))
	print('max ='+str(np.max(y)))
	print('std ='+str(np.std(y)))
	print('mean='+str(np.mean(y)))
	print('time='+str(t1-t0)+'s')
def test2D(x,y,a,b,GPU):
	testFilt=Filter(x,y,1,a,b,False)
	x=np.random.randint(a,b,(y,x))
	t0=time.clock()
	y=testFilt.insert2D(x)
	t1=time.clock()
	print(x)
	print(y)
	print('min ='+str(np.min(y)))
	print('max ='+str(np.max(y)))
	print('std ='+str(np.std(y)))
	print('mean='+str(np.mean(y)))
	print('time='+str(t1-t0)+'s')

#test2D(4,6,0,255,True)
#test2D(4,6,0,255,False)
#test2D(8,4,123,456,False)
#test2D(8,4,123,456,True)
#test2D(5234,5678,100,2893,True)
#test2D(5234,5678,100,2893,False)

#test1D(6,0,255,True)
#test1D(6,0,255,False)
#test1D(8,123,456,False)
#test1D(8,123,456,True)
#test1D(5678,100,2893,True)
#test1D(5678,100,2893,False)
