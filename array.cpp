#include"array.h"

Array::Array(int n){
	int i;
	this->n=n;
	item=new float[n];
	for(i=0;i<n;i++){
		item[i]=0;
	}
}
Array::Array(const float *x,const int n){
	int i;
	this->n=n;
	this->item=new float[n];
	if(x){
		for(i=0;i<n;i++){
			this->item[i]=x[i];
		}
//			memcpy(this->item,x,n*sizeof(float));
	}
}
Array::~Array(){
	delete item;
}
void Array::print(){
	int i;
	float *x;
	x=this->item;
	for(i=0;i<this->n;i++){
		printf("[%i]%.02f\t",i,*x++);
	}
	printf("\n");
}
void Array::rand(){
	int i;
	for(i=0;i<this->n;i++){
		this->item[i]=(random()-(RAND_MAX/2))*2.0/((float)RAND_MAX)/((float)RANDSCALING);
	}
}
