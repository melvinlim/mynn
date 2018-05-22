#ifndef _XOR_H
#define _XOR_H
#include"array.h"
#include"data.h"
#include<time.h>

const double ex1[2]={-1,-1};
const double ex2[2]={-1,+1};
const double ex3[2]={+1,-1};
const double ex4[2]={+1,+1};
const double ans1[2]={-1,+1};
const double ans2[2]={+1,-1};
const double ans3[2]={+1,-1};
const double ans4[2]={-1,+1};

class XorData:public Data{
public:
	XorData();
	~XorData();
	void status(Array<double> **,const Array<double> *,const Array<double> *);
};
#endif
