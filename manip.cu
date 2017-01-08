#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "types.h"

#define RANDSCALING 10	//scale random weights to be from -0.1 to +0.1

void PRINTMATRIX(Matrix *M){
	int i,j;
	for(i=0;i<M->height;i++){
		for(j=0;j<M->width;j++){
			printf("[%i,%i]%.09f ",i,j,M->elements[i*M->stride+j]);
		}
		printf("\n");
	}
	printf("\n");
}
void PRINTARRAY(Array *A){
	int i;
	float *x;
	int sz=A->len;
	x=A->el;
	for(i=0;i<sz;i++){
		printf("[%i]%.02f\t",i,*x++);
	}
	printf("\n");
}
void randMatrix(Matrix *M){
	int i,j;
	for(i=0;i<M->height;i++){
		for(j=0;j<M->width;j++){
			M->elements[i*M->stride+j]=
			(random()-(RAND_MAX/2))*2.0/((float)RAND_MAX)/((float)RANDSCALING);
		}
	}
}
void randArray(Array *A){
	int i;
	for(i=0;i<A->len;i++){
		A->el[i]=
		(random()-(RAND_MAX/2))*2.0/((float)RAND_MAX)/((float)RANDSCALING);
	}
}
void nnRand(Net *N){
	Matrix *pM;
	int i;
	int n=N->size;
	for(i=0;i<n;i++){
		pM=N->L[i]->M;
		randMatrix(pM);
	}
}
Array *CREATEARRAY(const float *x,int n){
	Array *p=(Array *)malloc(sizeof(Array));
	p->len=n;
	p->el=(float *)malloc(n*sizeof(float));
	if(x){
		memcpy(p->el,x,n*sizeof(float));
	}
	return(p);
}
