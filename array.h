#ifndef _ARRAY_H
#define _ARRAY_H
#include<stdio.h>
#include<stdlib.h>
#include<cstdint>
#include"defs.h"

class Array{
public:
	int n;
	double *item;
	Array(int n){
		int i;
		this->n=n;
		item=new double[n];
		for(i=0;i<n;i++){
			item[i]=0;
		}
	}
	Array(const double *x,const int n){
		int i;
		this->n=n;
		this->item=new double[n];
		if(x){
			for(i=0;i<n;i++){
				this->item[i]=x[i];
			}
		}
	}
	virtual ~Array(){
		if(item)
			delete[] item;
	}
	virtual void print(){
		int i;
		double *x;
		x=this->item;
		for(i=0;i<this->n;i++){
			printf("[%3i] %+.02f\t",i,*x++);
		}
		printf("\n");
	}
	void randomize(){
		int i;
		for(i=0;i<this->n;i++){
			this->item[i]=(random()-(RAND_MAX/2))*2.0/((double)RAND_MAX)/((double)RANDSCALING);
		}
	}
};
#endif
