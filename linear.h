#ifndef _LINEAR_H
#define _LINEAR_H
#include"array.h"
#include"data.h"
#include<time.h>

const double exlinear1[3]={-1,-1,+1};
const double exlinear2[3]={-1,+1,+1};
const double exlinear3[3]={+1,-1,+1};
const double exlinear4[3]={+1,+1,+1};
const double exlinear5[3]={-1,-1,-1};
const double exlinear6[3]={-1,+1,-1};
const double exlinear7[3]={+1,-1,-1};
const double exlinear8[3]={+1,+1,-1};
const double anslinear1[1]={-50};
const double anslinear2[1]={+50};
const double anslinear3[1]={+0.05};
const double anslinear4[1]={-0.05};
const double anslinear5[1]={-25};
const double anslinear6[1]={+35};
const double anslinear7[1]={+5};
const double anslinear8[1]={-5};

class LinearData:public Data{
public:
	LinearData();
	~LinearData();
	void status(Array<double> **,const Array<double> *,const Array<double> *);
};
#endif
