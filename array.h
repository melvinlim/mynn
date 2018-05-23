#ifndef _ARRAY_H
#define _ARRAY_H
#include<stdio.h>
#include<stdlib.h>
#include<cstdint>

template<typename T>
class Array{
public:
	int nElements;
	T *item;
	Array(int n){
		nElements=n;
		item=new T[nElements]();
	}
	Array(const T *x,const T *y,const int xn,const int yn){
		int i;
		nElements=xn+yn;
		item=new T[nElements];
		int p=0;
		if(x){
			for(i=0;i<xn;i++){
				item[p++]=x[i];
			}
			for(i=0;i<yn;i++){
				item[p++]=y[i];
			}
		}
	}
	Array(const T *x,const int n){
		int i;
		nElements=n;
		item=new T[n];
		if(x){
			for(i=0;i<n;i++){
				item[i]=x[i];
			}
		}
	}
	virtual ~Array(){
		if(item)
			delete[] item;
	}
	virtual void print(int rowSize=0) const{
		int i;
		T *x;
		x=item;
		if(rowSize==0)	rowSize=nElements;
		for(i=0;i<nElements;i++){
			printf("[%3i] %+.02f\t",i,*x++);
			if(i%rowSize==0)	printf("\n");
		}
		printf("\n");
	}
	virtual void print(int rowSize=0){
		int i;
		T *x;
		x=item;
		if(rowSize==0)	rowSize=nElements;
		for(i=0;i<nElements;i++){
			if(i%rowSize==0)	printf("\n");
			printf("[%3i] %+.02f\t",i,*x++);
		}
		printf("\n");
	}
	void randomize(double scale_factor){
		int i;
		for(i=0;i<nElements;i++){
			item[i]=(random()-(RAND_MAX/2))*2.0/((double)RAND_MAX)/((double)scale_factor);
		}
	}
	T &operator[](int i){
		return item[i];
	}
	const T &operator[](int i) const{
		return item[i];
	}
	T &operator()(int i){
		return item[i];
	}
	const T &operator()(int i) const{
		return item[i];
	}
	friend bool operator==(const Array &lhs,const Array &rhs){
		T tol=0.0001;
		int n=lhs.nElements;
		for(int i=0;i<n;i++){
			if(fabs(lhs.item[i]-rhs.item[i])>tol){
				return false;
			}
		}
		return true;
	}
};
#endif
