#ifndef _DATA_H
#define _DATA_H
#include"array.h"
#include<time.h>
#include<assert.h>

class Data{
protected:
	Array **arrays;
	Array **pInputs;
	Array **pOutputs;
public:
	int nOutputs;
	int sz;
	int index;
	Data();
	~Data();
	Array **fillIOArrays(bool=false);
	double sumSqError(const Array *);
	int toLabel(const double *);
};
#endif
