import numpy as np
from csvWrap import *
from filt import *
import extract

def generate(target='xor'):
	if target=='xor':
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

		return [inp,out]

	elif target=='linearxor':
		inp1=np.array([-1,-1]).astype(np.float64)
		inp2=np.array([-1,+1]).astype(np.float64)
		inp3=np.array([+1,-1]).astype(np.float64)
		inp4=np.array([+1,+1]).astype(np.float64)
		inp=[inp1,inp2,inp3,inp4]

		out1=np.array([-2,-2]).astype(np.float64)
		out2=np.array([-1.5,+1.5]).astype(np.float64)
		out3=np.array([+1.5,-1.5]).astype(np.float64)
		out4=np.array([+2,+2]).astype(np.float64)
		out=[out1,out2,out3,out4]

		return [inp,out]

	elif target=='csv':
		#NEXAMPLES=30
		NEXAMPLES=4
		NPOS=NEXAMPLES
		NNEG=NEXAMPLES

		inp=[]
		out=[]

		#INPUTS=129600
		#filt=Filter(1,INPUTS,1,0,255,False)
		INPUTS=129600/3
		filt=Filter(1,INPUTS,1,0,255*3,False)

		for i in range(NPOS):
			filename='verified/pos'+str(i)+'.csv'
			x=readCSV(filename)
			t=np.array(map(int,x)).astype(np.float64)
			t=t.reshape(-1,len(t)/3,3).sum(axis=2)[0]
			y=filt.insert1D(t)
			inp.append(y)
			t=np.array([-1,+1]).astype(np.float64)
			out.append(t)

		for i in range(NNEG):
			filename='verified/neg'+str(i)+'.csv'
			x=readCSV(filename)
			t=np.array(map(int,x)).astype(np.float64)
			t=t.reshape(-1,len(t)/3,3).sum(axis=2)[0]
			y=filt.insert1D(t)
			inp.append(y)
			t=np.array([+1,-1]).astype(np.float64)
			out.append(t)
		return [inp,out]

	elif target=='mnist':
		INPUTS=28*28
		filt=Filter(1,INPUTS,1,0,255,False)
		tmp=extract.extract()
		[[testLabel,testSet],[trainLabel,trainSet]]=tmp
		inp=[]
		out=[]
		for i in testSet:
			t=np.array(i).astype(np.float64)
			y=filt.insert1D(t)
			inp.append(y)
		for i in testLabel:
			tmp=[]
			for j in range(i):
				tmp.append(-1)
		#	tmp.append(1)
		#	for j in range(10-i-1):
		#		tmp.append(-1)
			for j in range(10-i):
				tmp.append(1)
			out.append(np.array(tmp).astype(np.float64))
		return [inp,out]
