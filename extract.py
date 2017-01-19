def read32(fp):
	val=fp.read(4)
	res=[]
	for i in val:
		res.append(ord(i))
	return res
def list32ToValue(x):
	ret=0
	for i in range(4):
		ret+=x[i]*(256**(4-i-1))
	return ret
def readImage(filename,magicNumber,number,rows,cols):
	fp=open(filename)
	res=read32(fp)
	assert(res==magicNumber)
	res=read32(fp)
	assert(res==number)
	n=list32ToValue(res)

	res=read32(fp)
	assert(res==rows)
	res=read32(fp)
	assert(res==cols)
	arr=[]
	for i in range(n):
		val=fp.read(1)
		arr.append(ord(val))
	return arr
def readLabel(filename,magicNumber,number):
	fp=open(filename)
	res=read32(fp)
	assert(res==magicNumber)
	res=read32(fp)
	assert(res==number)
	n=list32ToValue(res)
	arr=[]
	for i in range(n):
		val=fp.read(1)
		arr.append(ord(val))
	return arr
def extract():
	tl=readLabel('t10k-labels-idx1-ubyte',[0,0,8,1],[0,0,0x27,0x10])
	ti=readImage('t10k-images-idx3-ubyte',[0,0,8,3],[0,0,0x27,0x10],[0,0,0,0x1c],[0,0,0,0x1c])
	l=readLabel('train-labels-idx1-ubyte',[0,0,8,1],[0,0,0xea,0x60])
	i=readImage('train-images-idx3-ubyte',[0,0,8,3],[0,0,0xea,0x60],[0,0,0,0x1c],[0,0,0,0x1c])
	return [ [tl,ti],[l,i] ]
arr=extract()