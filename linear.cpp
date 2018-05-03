#include"linear.h"

LinearData::~LinearData(){}
LinearData::LinearData():Data(){
	sz=4;
	nOutputs=4;
	pInputs=new Array<double> *[sz];
	pOutputs=new Array<double> *[sz];
	pInputs[0]=new Array<double>(exlinear1,NINPUTS);
	pInputs[1]=new Array<double>(exlinear2,NINPUTS);
	pInputs[2]=new Array<double>(exlinear3,NINPUTS);
	pInputs[3]=new Array<double>(exlinear4,NINPUTS);
	pOutputs[0]=new Array<double>(anslinear1,NOUTPUTS);
	pOutputs[1]=new Array<double>(anslinear2,NOUTPUTS);
	pOutputs[2]=new Array<double>(anslinear3,NOUTPUTS);
	pOutputs[3]=new Array<double>(anslinear4,NOUTPUTS);
}
void LinearData::status(Array<double> **ioArrays,const Array<double> *response,const Array<double> *error){
	Array<double> *pIn=ioArrays[0];
	Array<double> *pOut=ioArrays[1];
	printf("in:[%+.0f,%+.0f] resp:[%+.2f,%+.2f] targ:[%+.2f,%+.2f] err:[%+.2f,%+.2f]\n",
	pIn->item[0],pIn->item[1],
	response->item[0],response->item[1],
	pOut->item[0],pOut->item[1],
	error->item[0],error->item[1]
	);
}
