#ifndef _LAYER_H
#define _LAYER_H
#include<assert.h>
#include<math.h>
#include"defs.h"
#include"matrix.h"
#include"array.h"
class Layer{
public:
	int nRows;
	int nCols;
	Matrix *mat;
	Matrix *dw;
	Array *out;
	Array *deriv;
	Array *delta;
	Layer(int,int);
	Layer(Matrix *);
	~Layer();
	Array *forward(const Array *);
	void outputDelta(const Array *);
	void hiddenDelta(const Matrix *,const Array *);
	void saveErrors(const Array *);
	void updateWeights();
	void directUpdateWeights(const Array *);
	void randomize();
};
#endif
