#ifndef _IDX_H
#define _IDX_H
#include<stdio.h>
#include<cstdint>
#include<assert.h>
#include<fcntl.h>
#include<unistd.h>
#include<sys/mman.h>
#include"matrix.h"
#include"net.h"
#include"string.h"
#include"defs.h"
namespace IDX{
	void displayImage(double *img);
	void printImage(struct image *img);
	Matrix<double> *loadIDXEntry(struct idx2 *);
	void saveIDX(Matrix<double> &,const char *);
	void saveIDXEntry(Matrix<double> &,int);
	Matrix<double> *loadIDX(const char *);
	void saveNetwork(Net *,const char *);
	Net *loadNetwork(const char *);
	bool verifiedHeader(struct idx2 *);
};
#endif
