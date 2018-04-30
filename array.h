#ifndef _ARRAY_H
#define _ARRAY_H
#include<stdio.h>
#include<stdlib.h>
#include"defs.h"

class Array{
public:
	int n;
	float *item;
	Array(int n);
	Array(const float *x,const int n);
	~Array();
	void print();
	void rand();
};
#endif
