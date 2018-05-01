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
	Net(int,int);
	~Net();
	void insertLayer(int,int,int);
	void insertLayer(int,Matrix *);
	void forward(const Array *);
	inline void backward();
	void randomize();
	void print();
	Array *trainBatch(const Array *,const Array *);
	Array *trainOnce(const Array *,const Array *);
	void updateError(const Array *);
	inline void updateBatchCorrections(const Array *);
	inline void directUpdateWeights(const Array *);
	void updateWeights();
};

class SingleHidden:public Net{
public:
	SingleHidden(int,int,int);
};
#endif
