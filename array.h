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
	Array(int n);
	Array(const double *x,const int n);
	~Array();
	void print();
	void rand();
};
class NetArray:public Array{
public:
	NetArray(uint8_t,int);
	NetArray(uint8_t *,int);
	~NetArray();
};
#endif
