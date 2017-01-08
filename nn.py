import numpy as np

class Layer:
	def __init__(self,n,m):
		self.A=np.random.randint(-10000,10000,(n,m))/100000.0
		self.out=np.array(n)
	def insert(self,x):
		self.out=np.dot(self.A,x)
		return self.out

#def forward(A1,A2,

NN=[Layer(16,2),Layer(2,16)]
#A1=np.random.randint(-10000,10000,(16,2))/100000.0
#A2=np.random.randint(-10000,10000,(2,16))/100000.0
inp1=np.array([-1,-1])
inp2=np.array([-1,+1])
inp3=np.array([+1,-1])
inp4=np.array([+1,+1])
inp=[inp1,inp2,inp3,inp4]
out1=np.array([-1,-1])
out2=np.array([-1,+1])
out3=np.array([+1,-1])
out4=np.array([+1,+1])
out=[out1,out2,out3,out4]

tmp=NN[0].insert(inp1)
tmp=NN[1].insert(tmp)
tmp
