#ifndef _IDX_H
#define _IDX_H
#include<stdio.h>
#include<cstdint>
#include<assert.h>
#include<fcntl.h>
#include<unistd.h>
#include"matrix.h"
#include"string.h"
#include"defs.h"
namespace IDX{
	void displayImage(double *img);
	void printImage(struct image *img);
	void saveIDX(Matrix *,const char *);
	void loadIDX(Matrix *,const char *);
};
#endif
