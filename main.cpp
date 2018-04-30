#include"net.h"
#include"data.h"
#include"defs.h"

void displayImage(double *img){
	double *p=img;
	for(int i=0;i<28;i++){
		for(int j=0;j<28;j++){
			if(*p++>0){
				printf(".");
			}else{
				printf(" ");
			}
		}
		printf("\n");
	}
}

int main(){
	int i;
	Net *net=new SingleHidden(NINPUTS,HIDDEN,NOUTPUTS);
/*
	net->print();
	net->rand();
	net->print();
*/
	Array **arrays;
	Array *pIn,*pOut;

	MNISTData data;
	//XorData data;

	for(int i=0;i<8;i++){
		printf("%d:\n",i);
		arrays=data.fillIOArrays(true);
		arrays[1]->print();
		displayImage(arrays[0]->item);
		printf("\n");
	}

	for(i=0;i<EPOCHS;i++){
/*
Net *asdf=new Net(10);
for(int j=0;j<10;j++){
	asdf->insertLayer(j,10,10);
}
delete asdf;
*/
		arrays=data.fillIOArrays();
		pIn=arrays[0];
		pOut=arrays[1];
		net->train(pIn,pOut);
		if(i%10){
			net->updateWeights();
		}
	}
}
