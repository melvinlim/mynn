import numpy as np
EPOCHS=1000
GAMMA=0.1
class Layer:
	def __init__(self,n,m):
		self.A=np.random.randint(-10000,10000,(n,m))/100000.0
		self.out=np.array(n*[0])
		self.delta=np.array(n*[0])
		self.deriv=np.array(n*[0])
	def insert(self,x):
		self.out=np.tanh(np.dot(self.A,x))
		self.deriv=1.0-(self.out*self.out)
		return self.out
	def updateDelta0(self,y):
		self.delta=self.deriv*y
		return self.delta
	def updateDelta(self,A,y):
		arr=[]
		for j in range(len(self.delta)):
			s=0
			for k in range(len(y)):
				s += A[k][j]*y[k]
			arr.append(s)
		self.delta=self.deriv*s
#			self.delta[j]=self.deriv[j]*s
		return self.delta
	def updateWeights(self,x):
		for i in range(self.A.shape[0]):
			for j in range(self.A.shape[1]):
				self.A += self.delta[i]*x[j]
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

for i in range(EPOCHS):
	r=np.random.randint(0,4)
	tmp=NN[0].insert(inp[r])
	tmp=NN[1].insert(tmp)
	error=out[r]-tmp
	print(error)
	tmp=NN[1].updateDelta0(error)
	tmp=NN[0].updateDelta(NN[1].A,tmp)
	#print(tmp)
	NN[1].updateWeights(NN[0].out)
	NN[0].updateWeights(inp[r])
