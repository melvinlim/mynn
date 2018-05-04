#ifndef _LAYER_H
#define _LAYER_H
#include<assert.h>
#include<math.h>
#include"defs.h"
#include"matrix.h"
#include"array.h"
class Layer{
public:
	double gamma;
	int nRows;
	int nCols;
	Matrix<double> mat;
	Matrix<double> dw;
#ifdef TESTGRAD
	Matrix<double> dgw;
#endif
	Array<double> out;
	Array<double> deriv;
	Array<double> delta;
	Layer(int,int,double);
	Layer(Matrix<double> &,double);
	virtual ~Layer();
	virtual Array<double> &forward(const Array<double> &);
	void outputDelta(const Array<double> &);
	void hiddenDelta(const Matrix<double> &,const Array<double> &);
	void saveErrors(const Array<double> &);
	void updateWeights();
	void directUpdateWeights(const Array<double> &);
	void randomize();
};
class LinearLayer:public Layer{
public:
	LinearLayer(int a,int b,double c):Layer(a,b,c){};
	LinearLayer(Matrix<double> & a,double b):Layer(a,b){};
	~LinearLayer(){};
	Array<double> &forward(const Array<double> &);
};
#endif
