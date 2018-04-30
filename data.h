#ifndef _DATA_H
#define _DATA_H
#define NINPUTS (2)
#define NOUTPUTS (2)
#include"array.h"

const double ex1[NINPUTS]={-1,-1};
const double ex2[NINPUTS]={-1,+1};
const double ex3[NINPUTS]={+1,-1};
const double ex4[NINPUTS]={+1,+1};
const double ans1[NOUTPUTS]={-1,+1};
const double ans2[NOUTPUTS]={+1,-1};
const double ans3[NOUTPUTS]={+1,-1};
const double ans4[NOUTPUTS]={-1,+1};

class Data{
	Array **pInputs;
	Array **pOutputs;
public:
	Array *pIn,*pOut;
	int sz;
	int index;
	Data();
	~Data();
	void fillIOArrays(Array &,Array &);
};
#endif
