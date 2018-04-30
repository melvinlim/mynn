#ifndef _MATRIX_H
#define _MATRIX_H
#include<stdio.h>
#include<stdlib.h>
#include"defs.h"

class Matrix{
public:
	int n;
	int m;
	float *item;
	Matrix(int n,int m);
	~Matrix();
	float atIndex(int i,int j);
	void rand();
	void print();
};
#endif
