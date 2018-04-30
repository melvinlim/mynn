#ifndef _LAYER_H
#define _LAYER_H
#include<math.h>
#include"defs.h"
#include"matrix.h"
#include"array.h"
class Layer{
public:
	Matrix *M;
	Array *out;
	Array *deriv;
	Array *delta;
	Layer(int n,int m);
	~Layer();
	Array *forward(const Array *x);
	void outputDelta(const Array *error);
	void upDelta(const Matrix *W,const Array *delta2);
	void updateWeights(const Array *input);
	void rand();
};
#endif
