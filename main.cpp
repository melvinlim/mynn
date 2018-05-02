#include"net.h"
#include"data.h"
#include"defs.h"
#include"idx.h"
#include"mnist.h"
#include"xor.h"

int main(){
	time_t startTime,endTime;
	int i;
	Net *net;
	double sumSqErr;
	Array **arrays;
	Array *pIn,*pOut;
#ifdef SOLVEXOR
	XorData trainingData;
	XorData testingData;
#else
	MNISTTrainingData trainingData;
	MNISTTestingData testingData;
#endif
	time(&startTime);

	for(int i=0;i<8;i++){
		arrays=trainingData.fillIOArrays(true);
		printf("%d:\n",i);
		arrays[1]->print();
		arrays[0]->print();
	}
	int hidden=HIDDEN;
	double gamma=GAMMA;
	for(int network=0;network<4;network++){
		net=new SingleHidden(NINPUTS,hidden++,NOUTPUTS,gamma);
		sumSqErr=0;
		for(i=0;i<EPOCHS;i++){
			arrays=trainingData.fillIOArrays();
			pIn=arrays[0];
			pOut=arrays[1];
#ifdef BATCH
			net->trainBatch(pIn,pOut);
			sumSqErr+=trainingData.sumSqError(&net->error);
			if(i%4){
				net->updateWeights();
			}
#else
			net->trainOnce(pIn,pOut);
			sumSqErr+=trainingData.sumSqError(&net->error);
			printf("epoch: %i\n",i);
			trainingData.status(arrays,net->response,net->error);
#endif
		}
		printf("net: %d\n",network);
		printf("avg sse: %f\n",sumSqErr/(double)EPOCHS);
		for(int i=0;i<testingData.nOutputs;i++){
			arrays=testingData.fillIOArrays();
			pIn=arrays[0];
			pOut=arrays[1];
			net->forward(pIn);
			net->updateError(pOut);
			testingData.status(arrays,net->response,&net->error);
		}
/*
		IDX::saveNetwork(net,"test.idx");
		Net *tmpNet=IDX::loadNetwork("test.idx");
		assert(tmpNet->L[0]->mat==net->L[0]->mat);
		assert(tmpNet->L[1]->mat==net->L[1]->mat);
		delete tmpNet;
*/
		delete net;
	}
	time(&endTime);
	printf("%d seconds elapsed.\n",(int)difftime(endTime,startTime));
}
