#ifndef _NET_H
#define _NET_H

#include"defs.h"
#include"matrix.h"
#include"array.h"
#include"layer.h"

class Net{
public:
	Layer **L;
	int n;
	Array *error;
	Array *answer;
	Net(int n=0);
	~Net();
	void insertLayer(int i,int n,int m);
	void forward(const Array *x);
	void backward(const Array *input);
	void rand();
	void print();
	Array *train(const Array *x,const Array *y);
	void updateError(const Array *yTarget);
	void status(const Array *,const Array *);
};

#endif
/*
float nnTotalError(const Array *y0,const Array *y){
	int i;
	int n=y0->n;
	float ret=0;
	for(i=0;i<n;i++){
		ret+=fabs(y0->item[i]-y->item[i]);
		ret*=ret;
	}
	return(ret/2.0);
}
*/
