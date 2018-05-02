#include"xor.h"

XorData::~XorData(){}
XorData::XorData():Data(){
	sz=4;
	nOutputs=4;
	pInputs=new Array<double> *[sz];
	pOutputs=new Array<double> *[sz];
	pInputs[0]=new Array<double>(ex1,NINPUTS);
	pInputs[1]=new Array<double>(ex2,NINPUTS);
	pInputs[2]=new Array<double>(ex3,NINPUTS);
	pInputs[3]=new Array<double>(ex4,NINPUTS);
	pOutputs[0]=new Array<double>(ans1,NOUTPUTS);
	pOutputs[1]=new Array<double>(ans2,NOUTPUTS);
	pOutputs[2]=new Array<double>(ans3,NOUTPUTS);
	pOutputs[3]=new Array<double>(ans4,NOUTPUTS);
}
void XorData::status(Array<double> **ioArrays,const Array<double> *response,const Array<double> *error){
	Array<double> *pIn=ioArrays[0];
	Array<double> *pOut=ioArrays[1];
	printf("in:[%.0f,%.0f] resp:[%f,%f] targ:[%.0f,%.0f] err:[%f,%f]\n",
	pIn->item[0],pIn->item[1],
	response->item[0],response->item[1],
	pOut->item[0],pOut->item[1],
	error->item[0],error->item[1]
	);
}
