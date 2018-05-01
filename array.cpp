#include"array.h"

Array::Array(int n){
	int i;
	this->n=n;
	item=new double[n];
	for(i=0;i<n;i++){
		item[i]=0;
	}
}
Array::Array(const double *x,const int n){
	int i;
	this->n=n;
	this->item=new double[n];
	if(x){
		for(i=0;i<n;i++){
			this->item[i]=x[i];
		}
//			memcpy(this->item,x,n*sizeof(double));
	}
}
Array::~Array(){
	delete[] item;
}
void Array::print(){
	int i;
	double *x;
	x=this->item;
	for(i=0;i<this->n;i++){
		printf("[%3i] %+.02f\t",i,*x++);
	}
	printf("\n");
}
void Array::rand(){
	int i;
	for(i=0;i<this->n;i++){
		this->item[i]=(random()-(RAND_MAX/2))*2.0/((double)RAND_MAX)/((double)RANDSCALING);
	}
}
NetArray::NetArray(uint8_t label,int n):Array(n){
	for(int i=0;i<n;i++){
		item[i]=-1;
	}
	item[label]=1;
}
NetArray::NetArray(uint8_t *pixels,int n):Array(n){
	uint8_t *p=pixels;
	for(int i=0;i<n;i++){
		if(*p++>=127){
			item[i]=1;
		}else{
			item[i]=-1;
		}
	}
}
