#ifndef _MNIST_H
#define _MNIST_H
#include<stdio.h>
#include<stdlib.h>
#include<cstdint>
#include"array.h"
#include"defs.h"
#include"data.h"
#include"idx.h"

class MNISTArray:public Array{
public:
	MNISTArray(uint8_t *,int);
	~MNISTArray();
	void print();
};

class MNISTData:public Data{
public:
	MNISTData();
	~MNISTData();
	void status(Array **,const Array *,const Array *);
	Array *expandLabelArray(uint8_t,int);
};
#endif
