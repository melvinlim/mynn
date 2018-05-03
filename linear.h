#ifndef _LINEAR_H
#define _LINEAR_H
#include"array.h"
#include"data.h"
#include<time.h>

const double exlinear1[NINPUTS]={-1,-1};
const double exlinear2[NINPUTS]={-1,+1};
const double exlinear3[NINPUTS]={+1,-1};
const double exlinear4[NINPUTS]={+1,+1};
const double anslinear1[NOUTPUTS]={-50,23};
const double anslinear2[NOUTPUTS]={+50,-4};
const double anslinear3[NOUTPUTS]={+0.05,-10};
const double anslinear4[NOUTPUTS]={-0.05,40};

class LinearData:public Data{
public:
	LinearData();
	~LinearData();
	void status(Array<double> **,const Array<double> *,const Array<double> *);
};
#endif
