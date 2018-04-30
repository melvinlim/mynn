#ifndef _MATRIX_H
#define _MATRIX_H
#include<stdio.h>
#include<stdlib.h>
#include"defs.h"

class Matrix{
public:
	int m;
	int n;
	double *item;
	Matrix(int,int);
	~Matrix();
	double atIndex(int,int) const;
	double atIndex(int,int);
	void rand();
	void print();
};
#endif
