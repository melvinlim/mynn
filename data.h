#ifndef _DATA_H
#define _DATA_H
#include"array.h"
#include<time.h>
#include<assert.h>
#include<fcntl.h>
#include<unistd.h>
#include<sys/mman.h>

const double ex1[NINPUTS]={-1,-1};
const double ex2[NINPUTS]={-1,+1};
const double ex3[NINPUTS]={+1,-1};
const double ex4[NINPUTS]={+1,+1};
const double ans1[NOUTPUTS]={-1,+1};
const double ans2[NOUTPUTS]={+1,-1};
const double ans3[NOUTPUTS]={+1,-1};
const double ans4[NOUTPUTS]={-1,+1};

class Data{
protected:
	Array **arrays;
	Array **pInputs;
	Array **pOutputs;
public:
	int sz;
	int index;
	Data();
	~Data();
	Array **fillIOArrays(bool=false);
};
class MNISTData:public Data{
public:
	MNISTData();
	~MNISTData();
};
class XorData:public Data{
public:
	XorData();
	~XorData();
};
#endif
