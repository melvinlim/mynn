#include"mnist.h"

MNISTArray::~MNISTArray(){
}
void MNISTArray::print(){
	IDX::displayImage(item);
}
MNISTArray::MNISTArray(uint8_t *pixels,int n):Array(n){
	uint8_t *p=pixels;
	for(int i=0;i<n;i++){
		if(*p++>=127){
			item[i]=1;
		}else{
			item[i]=-1;
		}
	}
}

void MNISTData::status(Array **ioArrays,const Array *response,const Array *error){
	Array *pOut=ioArrays[1];
	printf("resp:%d targ:%d ssqerr:%f\n",
	toLabel(response->item),
	toLabel(pOut->item),
	sumSqError(error)
	);
}
Array *MNISTData::expandLabelArray(uint8_t label,int n){
	Array *array=new Array(n);
	double *item=array->item;
	for(int i=0;i<n;i++){
		item[i]=-1;
	}
	item[label]=1;
	return array;
}
MNISTTrainingData::~MNISTTrainingData(){}
MNISTTrainingData::MNISTTrainingData():MNISTData("t10k-labels-idx1-ubyte","t10k-images-idx3-ubyte"){}
MNISTTestingData::~MNISTTestingData(){}
MNISTTestingData::MNISTTestingData():MNISTData("train-labels-idx1-ubyte","train-images-idx3-ubyte"){}
MNISTData::~MNISTData(){}
MNISTData::MNISTData(const char *file1,const char *file2):Data(){
	void *mem1,*mem2;
	struct idx1 *idx1Header;
	struct idx3 *idx3Header;
	struct image *pImage;
	uint8_t *pLabel;
	int fd1=open(file1,O_RDONLY);
	int fd3=open(file2,O_RDONLY);
	assert(fd1>=0);
	assert(fd3>=0);
	nOutputs=10;
	mem1=mmap(0,6*1024*1024,PROT_READ,MAP_FILE|MAP_SHARED,fd1,0);
	mem2=mmap(0,6*8*1024*1024,PROT_READ,MAP_FILE|MAP_SHARED,fd3,0);
	idx1Header=(struct idx1 *)mem1;
	idx3Header=(struct idx3 *)mem2;
	assert(idx1Header!=MAP_FAILED);
	assert(idx3Header!=MAP_FAILED);
	assert(bswap_32(idx1Header->magic)==0x801);
	assert(bswap_32(idx3Header->magic)==0x803);
	int nLabels=bswap_32(idx1Header->number);
	printf("0x%x\n%d\n",bswap_32(idx1Header->magic),bswap_32(idx1Header->number));
	printf("0x%x\n%d\n",bswap_32(idx3Header->magic),bswap_32(idx3Header->nImages));
	pLabel=(uint8_t *)(++idx1Header);
	pImage=(struct image *)(++idx3Header);
	sz=nLabels;
	pInputs=new Array *[sz];
	pOutputs=new Array *[sz];
	int nInputs=28*28;
	for(int i=0;i<nLabels;i++){
		pInputs[i]=new MNISTArray(pImage->pixel,nInputs);
		pOutputs[i]=expandLabelArray(*pLabel,10);
//		pOutputs[i]=new MNISTArray(*pLabel,10);
//		printf("label %d: %d\n",i,*pLabel);
//		IDX::printImage(pImage);
		pLabel++;
		pImage++;
	}
	close(fd1);
	close(fd3);
	assert(munmap(mem1,1024*1024)==0);
	assert(munmap(mem2,8*1024*1024)==0);
}
