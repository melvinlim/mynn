#ifndef _MNIST_H
#define _MNIST_H
#include<stdio.h>
#include<stdlib.h>
#include<cstdint>
#include"array.h"
#include"defs.h"
#include"idx.h"

class MNISTArray:public Array{
public:
	MNISTArray(uint8_t *,int);
	~MNISTArray();
	void print();
};
#endif
