#ifndef _LAYER_H
#define _LAYER_H
#include<assert.h>
#include<math.h>
#include"matrix.h"
#include"array.h"
class Layer{
public:
	double gamma;
	double lambda_decay;
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
	Layer(int,int,double,double);
	Layer(Matrix<double> &,double,double);
	virtual ~Layer();
	virtual Array<double> &forward(const Array<double> &);
	void outputDelta(const Array<double> &);
	void hiddenDelta(const Matrix<double> &,const Array<double> &);
	void saveErrors(const Array<double> &);
	void updateWeights();
	void directUpdateWeights(const Array<double> &);
	void randomize(double);
};
class LinearLayer:public Layer{
public:
	LinearLayer(int a,int b,double c,double d):Layer(a,b,c,d){};
	LinearLayer(Matrix<double> & a,double b,double c):Layer(a,b,c){};
	~LinearLayer(){};
	Array<double> &forward(const Array<double> &);
};
#endif
