#include"data.h"

Data::Data(){
	index=0;
	this->sz=4;
	srand(time(0));

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
	pIn=pInputs[0];
	pOut=pOutputs[0];
}
Data::~Data(){
	for(int i=0;i<sz;i++){
		delete pInputs[i];
		delete pOutputs[i];
	}
	delete[] pInputs;
	delete[] pOutputs;
}
void Data::fillIOArrays(Array &inputArray,Array &outputArray,const bool randomize){
	inputArray=pIn[index];
	outputArray=pOut[index];
	if(randomize){
		index=random()%sz;
	}else{
		index=(index+1)%sz;
	}
}
