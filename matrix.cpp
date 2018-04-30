#include"matrix.h"

Matrix::Matrix(int m,int n){
	int i;
	this->m=m;
	this->n=n;
	item=new double[m*n];
	for(i=0;i<m*n;i++){
		item[i]=0;
	}
}
Matrix::~Matrix(){
	delete[] item;
}
double Matrix::atIndex(int i,int j) const{
	//assert(((i+1)*(j+1))<=(n*m));
	return item[i*n+j];
}
double Matrix::atIndex(int i,int j){
	//assert(((i+1)*(j+1))<=(n*m));
	return item[i*n+j];
}
void Matrix::rand(){
	int i,j;
	double *p=item;
	for(i=0;i<this->m;i++){
		for(j=0;j<this->n;j++){
			*p++=(random()-(RAND_MAX/2))*2.0/((double)RAND_MAX)/((double)RANDSCALING);
		}
	}
}
void Matrix::print(){
	int i,j;
	for(i=0;i<this->m;i++){
		for(j=0;j<this->n;j++){
			printf("[%i,%i]%.09f ",i,j,this->atIndex(i,j));
		}
		printf("\n");
	}
	printf("\n");
}
