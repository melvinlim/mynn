#ifndef _NET_H
#define _NET_H
#include<time.h>
#include"defs.h"
#include"matrix.h"
#include"array.h"
#include"layer.h"

class Net{
public:
	Layer **L;
	int n;
	Array *error;
	Array *response;
	Net(int n=0);
	~Net();
	void insertLayer(int,int,int);
	void forward(const Array *);
	void backward(const Array *);
	void rand();
	void print();
	Array *train(const Array *,const Array *);
	void updateError(const Array *);
	void updateWeights();
	void status(const Array *,const Array *);
	void MNISTStatus(const Array *);
};

class SingleHidden:public Net{
public:
	SingleHidden(int,int,int);
};
#endif
