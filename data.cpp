#include"data.h"

/*
const double ex1[NINPUTS]={-1,-1};
const double ex2[NINPUTS]={-1,+1};
const double ex3[NINPUTS]={+1,-1};
const double ex4[NINPUTS]={+1,+1};
const double ans1[NOUTPUTS]={-1,+1};
const double ans2[NOUTPUTS]={+1,-1};
const double ans3[NOUTPUTS]={+1,-1};
const double ans4[NOUTPUTS]={-1,+1};
*/
Data::Data(){
	Array *p1,*p2,*p3,*p4;
	Array *pAns1,*pAns2,*pAns3,*pAns4;
	index=0;
	this->sz=4;

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
void Data::fillIOArrays(Array &inputArray,Array &outputArray){
	inputArray=(pIn[index]);
	outputArray=pOut[index];
	index=(index+1)%sz;
}
