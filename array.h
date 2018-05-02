#ifndef _ARRAY_H
#define _ARRAY_H
#include<stdio.h>
#include<stdlib.h>
#include<cstdint>
#include"defs.h"

template<typename T>
class Array{
public:
	int nElements;
	T *item;
	Array(int n){
		nElements=n;
		item=new T[nElements]();
	}
	Array(const T *x,const int n){
		int i;
		nElements=n;
		item=new T[n];
		if(x){
			for(i=0;i<n;i++){
				item[i]=x[i];
			}
		}
	}
	virtual ~Array(){
		if(item)
			delete[] item;
	}
	virtual void print(){
		int i;
		T *x;
		x=item;
		for(i=0;i<nElements;i++){
			printf("[%3i] %+.02f\t",i,*x++);
		}
		printf("\n");
	}
	void randomize(){
		int i;
		for(i=0;i<nElements;i++){
			item[i]=(random()-(RAND_MAX/2))*2.0/((double)RAND_MAX)/((double)RANDSCALING);
		}
	}
};
#endif
