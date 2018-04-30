#ifndef _ARRAY_H
#define _ARRAY_H
#include<stdio.h>
#include<stdlib.h>
#include"defs.h"

class Array{
public:
	int n;
	double *item;
	Array(int n);
	Array(const double *x,const int n);
	~Array();
	void print();
	void rand();
};
#endif
