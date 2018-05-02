#ifndef _XOR_H
#define _XOR_H
#include"array.h"
#include"data.h"
#include<time.h>

const double ex1[NINPUTS]={-1,-1};
const double ex2[NINPUTS]={-1,+1};
const double ex3[NINPUTS]={+1,-1};
const double ex4[NINPUTS]={+1,+1};
const double ans1[NOUTPUTS]={-1,+1};
const double ans2[NOUTPUTS]={+1,-1};
const double ans3[NOUTPUTS]={+1,-1};
const double ans4[NOUTPUTS]={-1,+1};

class XorData:public Data{
public:
	XorData();
	~XorData();
	void status(Array<double> **,const Array<double> *,const Array<double> *);
};
#endif
