#include"net.h"
#include"data.h"
#include"defs.h"
#include"idx.h"
#include"mnist.h"
#include"xor.h"

int main(){
	int i;
	Net *net;
	Array **arrays;
	Array *pIn,*pOut;
	net=new SingleHidden(NINPUTS,HIDDEN,NOUTPUTS);
#ifdef SOLVEXOR
	XorData trainingData;
	XorData testingData;
#else
	MNISTTrainingData trainingData;
	MNISTTestingData testingData;
#endif

	for(int i=0;i<8;i++){
		printf("%d:\n",i);
		arrays=trainingData.fillIOArrays(true);
		arrays[1]->print();
		arrays[0]->print();
	}
	int hidden=HIDDEN;
	for(int network=0;network<4;network++){
		delete net;
		net=new SingleHidden(NINPUTS,hidden++,NOUTPUTS);
		for(i=0;i<EPOCHS;i++){
			arrays=trainingData.fillIOArrays();
			pIn=arrays[0];
			pOut=arrays[1];
#ifdef BATCH
			net->trainBatch(pIn,pOut);
			if(i%4){
				net->updateWeights();
			}
#else
			net->trainOnce(pIn,pOut);
			printf("epoch: %i\n",i);
			trainingData.status(arrays,net->response,net->error);
#endif
		}
		printf("net: %d\n",network);
		for(int i=0;i<testingData.nOutputs;i++){
			arrays=testingData.fillIOArrays();
			pIn=arrays[0];
			net->forward(pIn);
			testingData.status(arrays,net->response,net->error);
		}
	}
}
