#ifndef _DATA_H
#define _DATA_H
#include"array.h"
#include<time.h>
#include<assert.h>
#include"defs.h"

class Data{
protected:
	Array<double> **arrays;
	Array<double> **pInputs;
	Array<double> **pOutputs;
public:
	int nOutputs;
	int sz;
	int index;
	Data();
	~Data();
	Array<double> **fillIOArrays(bool=false);
	double sumSqError(const Array<double> *);
	int toLabel(const double *);
};
#endif
