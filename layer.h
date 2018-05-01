#ifndef _LAYER_H
#define _LAYER_H
#include<assert.h>
#include<math.h>
#include"defs.h"
#include"matrix.h"
#include"array.h"
class Layer{
public:
	Matrix *mat;
	Matrix *dw;
	Array *out;
	Array *deriv;
	Array *delta;
	Layer(int n,int m);
	~Layer();
	Array *forward(const Array *x);
	void outputDelta(const Array *error);
	void hiddenDelta(const Matrix *W,const Array *delta2);
	void saveErrors(const Array *input);
	void updateWeights();
	void directUpdateWeights(const Array *input);
	void randomize();
};
#endif
