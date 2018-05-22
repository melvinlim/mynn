#ifndef _NET_H
#define _NET_H
#include<time.h>
#include"matrix.h"
#include"array.h"
#include"layer.h"

class Net{
public:
	Layer **L;
	int n;
	double sse;
	Array<double> error;
	Array<double> *response;
	Net(int,int);
	~Net();
	void insertLayer(int,int,int,double,double);
	void insertLayer(int,Matrix<double> &,double,double);
	void forward(const Array<double> *);
	inline void backward();
	void randomize(double);
	void print();
	void gradientDescent(const Array<double> *,const Array<double> *);
	Array<double> &trainBatch(const Array<double> *,const Array<double> *);
	Array<double> &trainOnce(const Array<double> *,const Array<double> *);
	void updateError(const Array<double> *);
	inline void updateBatchCorrections(const Array<double> *);
	inline void directUpdateWeights(const Array<double> *);
	void updateWeights();
};

class SingleHidden:public Net{
public:
	SingleHidden(int,int,int,double,double,double);
};

class SingleHiddenLinear:public Net{
public:
	SingleHiddenLinear(int,int,int,double,double,double);
};
#endif
