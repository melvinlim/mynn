#ifndef _LINEAR_H
#define _LINEAR_H
#include"array.h"
#include"data.h"
#include<time.h>

const double exlinear1[NINPUTS]={-1,-1,+1};
const double exlinear2[NINPUTS]={-1,+1,+1};
const double exlinear3[NINPUTS]={+1,-1,+1};
const double exlinear4[NINPUTS]={+1,+1,+1};
const double exlinear5[NINPUTS]={-1,-1,-1};
const double exlinear6[NINPUTS]={-1,+1,-1};
const double exlinear7[NINPUTS]={+1,-1,-1};
const double exlinear8[NINPUTS]={+1,+1,-1};
const double anslinear1[NOUTPUTS]={-50};
const double anslinear2[NOUTPUTS]={+50};
const double anslinear3[NOUTPUTS]={+0.05};
const double anslinear4[NOUTPUTS]={-0.05};
const double anslinear5[NOUTPUTS]={-25};
const double anslinear6[NOUTPUTS]={+35};
const double anslinear7[NOUTPUTS]={+5};
const double anslinear8[NOUTPUTS]={-5};

class LinearData:public Data{
public:
	LinearData();
	~LinearData();
	void status(Array<double> **,const Array<double> *,const Array<double> *);
};
#endif
