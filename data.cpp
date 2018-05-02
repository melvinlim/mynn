#include"data.h"

Data::Data(){
	nOutputs=0;
	index=0;
	sz=0;
	srandom(time(0));
	pInputs=0;
	pOutputs=0;
	arrays=new Array<double> *[2];
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
Array<double> **Data::fillIOArrays(const bool randomize){
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
double Data::sumSqError(const Array<double> *array){
	int i;
	int n=array->n;
	double *error=array->item;
	double ret=0;
	for(i=0;i<n;i++){
		ret+=error[i]*error[i];
	}
	return ret;
}
int Data::toLabel(const double *x){
	int i=0;
	for(i=0;i<10;i++){
		if(x[i]>0)	return i;
	}
	return i;
}
