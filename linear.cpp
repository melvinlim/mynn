#include"linear.h"

LinearData::~LinearData(){}
LinearData::LinearData():Data(){
	sz=8;
	nOutputs=8;
	pInputs=new Array<double> *[sz];
	pOutputs=new Array<double> *[sz];
	pInputs[0]=new Array<double>(exlinear1,NINPUTS);
	pInputs[1]=new Array<double>(exlinear2,NINPUTS);
	pInputs[2]=new Array<double>(exlinear3,NINPUTS);
	pInputs[3]=new Array<double>(exlinear4,NINPUTS);
	pInputs[4]=new Array<double>(exlinear5,NINPUTS);
	pInputs[5]=new Array<double>(exlinear6,NINPUTS);
	pInputs[6]=new Array<double>(exlinear7,NINPUTS);
	pInputs[7]=new Array<double>(exlinear8,NINPUTS);
	pOutputs[0]=new Array<double>(anslinear1,NOUTPUTS);
	pOutputs[1]=new Array<double>(anslinear2,NOUTPUTS);
	pOutputs[2]=new Array<double>(anslinear3,NOUTPUTS);
	pOutputs[3]=new Array<double>(anslinear4,NOUTPUTS);
	pOutputs[4]=new Array<double>(anslinear5,NOUTPUTS);
	pOutputs[5]=new Array<double>(anslinear6,NOUTPUTS);
	pOutputs[6]=new Array<double>(anslinear7,NOUTPUTS);
	pOutputs[7]=new Array<double>(anslinear8,NOUTPUTS);
}
void LinearData::status(Array<double> **ioArrays,const Array<double> *response,const Array<double> *error){
	Array<double> *pIn=ioArrays[0];
	Array<double> *pOut=ioArrays[1];
	printf("input:");
	pIn->print();
	printf("target:");
	pOut->print();
	printf("response:");
	response->print();
	printf("error:");
	error->print();
}
