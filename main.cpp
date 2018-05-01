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
	Net *net;
	Array **arrays;
	Array *pIn,*pOut;
	net=new SingleHidden(NINPUTS,HIDDEN,NOUTPUTS);
	net->randomize();
/*
	net=new Net(2);
	net->insertLayer(0,3,10);
	net->insertLayer(1,11,2);
	net->randomize();
*/
#ifdef SOLVEXOR
	XorData data;
#else
//	MNISTData data;
#endif

	for(int i=0;i<8;i++){
		printf("%d:\n",i);
		arrays=data.fillIOArrays(true);
		arrays[1]->print();
#ifdef SOLVEXOR
		arrays[0]->print();
#else
		displayImage(arrays[0]->item);
#endif
		printf("\n");
	}
/*
	int hidden=10;
	for(int network=0;network<100;network++){
		delete net;
//		net=new SingleHidden(NINPUTS,hidden++,NOUTPUTS);
		net=new Net(2);
		net->insertLayer(0,3,10);
		net->insertLayer(1,11,2);
		net->randomize();
*/
		for(i=0;i<EPOCHS;i++){
			arrays=data.fillIOArrays();
			pIn=arrays[0];
			pOut=arrays[1];
			net->train(pIn,pOut);
			if(i%4){
				net->updateWeights();
			}
		}
/*
		printf("net: %d\n",network);
		for(int i=0;i<4;i++){
			arrays=data.fillIOArrays();
			pIn=arrays[0];
			pOut=arrays[1];
			net->train(pIn,pOut);
			net->status(pIn,pOut);
		}
	}
*/
}
