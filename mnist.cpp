#include"mnist.h"

MNISTArray::~MNISTArray(){
}
void MNISTArray::print(){
	IDX::displayImage(item);
}
MNISTArray::MNISTArray(uint8_t *pixels,int n):Array(n){
	uint8_t *p=pixels;
	for(int i=0;i<n;i++){
		if(*p++>=127){
			item[i]=1;
		}else{
			item[i]=-1;
		}
	}
}
