#include"matrix.h"

Matrix::Matrix(int n,int m){
	int i;
	this->n=n;
	this->m=m;
	item=new float[n*m];
	for(i=0;i<n*m;i++){
		item[i]=0;
	}
}
Matrix::~Matrix(){
	delete item;
}
float Matrix::atIndex(int i,int j){
	//assert(((i+1)*(j+1))<=(n*m));
	return item[i*m+j];
}
void Matrix::rand(){
	int i,j;
	float *p=item;
	for(i=0;i<this->n;i++){
		for(j=0;j<this->m;j++){
			*p++=(random()-(RAND_MAX/2))*2.0/((float)RAND_MAX)/((float)RANDSCALING);
		}
	}
}
void Matrix::print(){
	int i,j;
	for(i=0;i<this->n;i++){
		for(j=0;j<this->m;j++){
			printf("[%i,%i]%.09f ",i,j,this->atIndex(i,j));
		}
		printf("\n");
	}
	printf("\n");
}
