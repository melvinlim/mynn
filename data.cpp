#include"data.h"

Data::Data(){
	Array *p1,*p2,*p3,*p4;
	Array *pAns1,*pAns2,*pAns3,*pAns4;
	index=0;
	this->sz=4;
	srand(time(0));

	p1=new Array(ex1,NINPUTS);
	p2=new Array(ex2,NINPUTS);
	p3=new Array(ex3,NINPUTS);
	p4=new Array(ex4,NINPUTS);

	pAns1=new Array(ans1,NOUTPUTS);
	pAns2=new Array(ans2,NOUTPUTS);
	pAns3=new Array(ans3,NOUTPUTS);
	pAns4=new Array(ans4,NOUTPUTS);
	
	pInputs=(Array **)malloc(4*sizeof(Array *));
	pInputs[0]=p1;
	pInputs[1]=p2;
	pInputs[2]=p3;
	pInputs[3]=p4;
	pOutputs=(Array **)malloc(4*sizeof(Array *));
	pOutputs[0]=pAns1;
	pOutputs[1]=pAns2;
	pOutputs[2]=pAns3;
	pOutputs[3]=pAns4;
	pIn=pInputs[0];
	pOut=pOutputs[0];
}
Data::~Data(){
}
void Data::fillIOArrays(Array &inputArray,Array &outputArray,const bool randomize){
	inputArray=(pIn[index]);
	outputArray=pOut[index];
	if(randomize){
		index=random()%sz;
	}else{
		index=(index+1)%sz;
	}
}
