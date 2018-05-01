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
int Data::toLabel(const double *x){
	int i=0;
	for(i=0;i<10;i++){
		if(x[i]>0)	return i;
	}
	return i;
}
