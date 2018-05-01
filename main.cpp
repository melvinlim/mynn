#include"net.h"
#include"data.h"
#include"defs.h"
#include"idx.h"

int main(){
	int i;
	Net *net;
	Array **arrays;
	Array *pIn,*pOut;
	net=new SingleHidden(NINPUTS,HIDDEN,NOUTPUTS);
#ifdef SOLVEXOR
	XorData data;
#else
	MNISTData data;
#endif

	for(int i=0;i<8;i++){
		printf("%d:\n",i);
		arrays=data.fillIOArrays(true);
		arrays[1]->print();
#ifdef SOLVEXOR
		arrays[0]->print();
#else
		IDX::displayImage(arrays[0]->item);
#endif
		printf("\n");
	}
	int hidden=15;
	for(int network=0;network<5;network++){
//		delete net;
		net=new SingleHidden(NINPUTS,hidden++,NOUTPUTS);
		for(i=0;i<EPOCHS;i++){
			arrays=data.fillIOArrays();
			pIn=arrays[0];
			pOut=arrays[1];
			net->train(pIn,pOut);
			if(i%4){
				net->updateWeights();
			}
		}
		printf("net: %d\n",network);
		for(int i=0;i<4;i++){
			arrays=data.fillIOArrays();
			pIn=arrays[0];
			pOut=arrays[1];
			net->forward(pIn);
			net->status(pIn,pOut);
		}
		IDX::saveIDX(net->L[0]->mat,"test.idx");
		Matrix *tmpMat=new Matrix(net->L[0]->mat->m,net->L[0]->mat->n);
		IDX::loadIDX(tmpMat,"test.idx");
	}
}
