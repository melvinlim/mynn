#include"data.h"

Data::Data(){
	nOutputs=0;
	index=0;
	sz=0;
	srandom(time(0));
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
	delete[] arrays;
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
	nOutputs=4;
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
	void *mem1,*mem2;
	struct idx1 *idx1Header;
	struct idx3 *idx3Header;
	struct image *pImage;
	uint8_t *pLabel;
	int fd1=open("t10k-labels-idx1-ubyte",O_RDONLY);
	int fd3=open("t10k-images-idx3-ubyte",O_RDONLY);
	assert(fd1>=0);
	assert(fd3>=0);
	nOutputs=10;
	mem1=mmap(0,1024*1024,PROT_READ,MAP_FILE|MAP_SHARED,fd1,0);
	mem2=mmap(0,8*1024*1024,PROT_READ,MAP_FILE|MAP_SHARED,fd3,0);
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
void XorData::status(Array **ioArrays,const Array *response,const Array *error){
	Array *pIn=ioArrays[0];
	Array *pOut=ioArrays[1];
	printf("in:[%.0f,%.0f] resp:[%f,%f] targ:[%.0f,%.0f] err:[%f,%f]\n",
	pIn->item[0],pIn->item[1],
	response->item[0],response->item[1],
	pOut->item[0],pOut->item[1],
	error->item[0],error->item[1]
	);
}
double Data::sumSqError(const Array *array){
	int i;
	int n=array->n;
	double *error=array->item;
	double ret=0;
	for(i=0;i<n;i++){
		ret+=error[i]*error[i];
	}
	return(ret/2.0);
}
int Data::toLabel(double *x){
	int i=0;
	for(i=0;i<10;i++){
		if(*x++>0)	return i;
	}
	return i;
}
void MNISTData::status(Array **ioArrays,const Array *response,const Array *error){
	Array *pOut=ioArrays[0];
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
