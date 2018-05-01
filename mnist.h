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
	MNISTData(const char *,const char *);
	~MNISTData();
	void status(Array **,const Array *,const Array *);
	Array *expandLabelArray(uint8_t,int);
};
class MNISTTrainingData:public MNISTData{
public:
	MNISTTrainingData();
	~MNISTTrainingData();
};
class MNISTTestingData:public MNISTData{
public:
	MNISTTestingData();
	~MNISTTestingData();
};
#endif
