#include"data.h"

Data::Data(){
	index=0;
	sz=0;
	srand(time(0));
	pInputs=0;
	pOutputs=0;
	arrays=new Array *[2];
}
Data::~Data(){
	for(int i=0;i<sz;i++){
		delete pInputs[i];
		delete pOutputs[i];
	}
	delete[] pInputs;
	delete[] pOutputs;
}
Array **Data::fillIOArrays(const bool randomize){
	assert(pInputs&&pOutputs);
	arrays[0]=pInputs[index];
	arrays[1]=pOutputs[index];
	if(randomize){
		index=random()%sz;
	}else{
		index=(index+1)%sz;
	}
	return arrays;
}
XorData::~XorData(){}
XorData::XorData():Data(){
	sz=4;
	pInputs=new Array *[sz];
	pOutputs=new Array *[sz];
	pInputs[0]=new Array(ex1,NINPUTS);
	pInputs[1]=new Array(ex2,NINPUTS);
	pInputs[2]=new Array(ex3,NINPUTS);
	pInputs[3]=new Array(ex4,NINPUTS);
	pOutputs[0]=new Array(ans1,NOUTPUTS);
	pOutputs[1]=new Array(ans2,NOUTPUTS);
	pOutputs[2]=new Array(ans3,NOUTPUTS);
	pOutputs[3]=new Array(ans4,NOUTPUTS);
}
MNISTData::~MNISTData(){}
MNISTData::MNISTData():Data(){
	struct idx1 *idx1Header;
	struct idx3 *idx3Header;
	int fd1=open("t10k-labels-idx1-ubyte",O_RDONLY);
	int fd3=open("t10k-images-idx3-ubyte",O_RDONLY);
	assert(fd1>=0);
	assert(fd3>=0);
	idx1Header=(struct idx1 *)mmap(0,1024*1024,PROT_READ,MAP_FILE|MAP_SHARED,fd1,0);
	idx3Header=(struct idx3 *)mmap(0,1024*1024,PROT_READ,MAP_FILE|MAP_SHARED,fd3,0);
	assert(idx1Header!=MAP_FAILED);
	assert(idx3Header!=MAP_FAILED);
	assert(bswap_32(idx1Header->magic)==0x801);
	assert(bswap_32(idx3Header->magic)==0x803);
	printf("0x%x\n%d\n",bswap_32(idx1Header->magic),bswap_32(idx1Header->number));
	printf("0x%x\n%d\n",bswap_32(idx3Header->magic),bswap_32(idx3Header->nImages));
}
