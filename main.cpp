#include"net.h"
#include"data.h"
#include"defs.h"
#include"idx.h"
#include"mnist.h"
#include"xor.h"
#include"linear.h"

int main(){
	time_t startTime,endTime;
	int i;
	Net *net;
	double sumSqErr;
	Array<double> **arrays;
	Array<double> *pIn,*pOut;
#ifdef SOLVEXOR
	XorData trainingData;
	XorData testingData;
#elif defined SOLVELINEAR
	LinearData trainingData;
	LinearData testingData;
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
	double lambda_decay=LAMBDA_DECAY;
	for(int network=0;network<4;network++){
		#ifdef SOLVELINEAR
			net=new SingleHiddenLinear(NINPUTS,hidden++,NOUTPUTS,gamma,lambda_decay,RANDSCALING);
		#else
			net=new SingleHidden(NINPUTS,hidden++,NOUTPUTS,gamma,lambda_decay,RANDSCALING);
		#endif
		for(i=0;i<EPOCHS;i++){
			arrays=trainingData.fillIOArrays();
			pIn=arrays[0];
			pOut=arrays[1];
#ifdef TESTGRAD
			net->gradientDescent(pIn,pOut);
#endif
#ifdef BATCH
			net->trainBatch(pIn,pOut);
#ifdef TESTGRAD
			assert(net->L[0]->dgw==net->L[0]->dw);
			assert(net->L[1]->dgw==net->L[1]->dw);
			if(true){
#else
			if(i%4){
#endif
				net->updateWeights();
			}
#else
			net->trainOnce(pIn,pOut);
			printf("epoch: %i\n",i);
			trainingData.status(arrays,net->response,&net->error);
#endif
		}
		sumSqErr=0;
		for(int i=0;i<testingData.nOutputs;i++){
			arrays=testingData.fillIOArrays();
			pIn=arrays[0];
			pOut=arrays[1];
			net->forward(pIn);
			net->updateError(pOut);
			sumSqErr+=testingData.sumSqError(&net->error);
			testingData.status(arrays,net->response,&net->error);
		}
		printf("net: %d\n",network);
		printf("avg sse: %f\n",sumSqErr/(double)testingData.nOutputs);
#ifdef TESTSAVELOAD
		IDX::saveNetwork(net,"test.idx");
		SingleHidden tmpNet(NINPUTS,hidden++,NOUTPUTS,gamma,lambda_decay,RANDSCALING);
		IDX::loadNetwork(&tmpNet,"test.idx",GAMMA,LAMBDA_DECAY);
		assert(tmpNet.L[0]->mat==net->L[0]->mat);
		assert(tmpNet.L[1]->mat==net->L[1]->mat);
		Net *pTmpNet=IDX::loadNetwork("test.idx",GAMMA,LAMBDA_DECAY);
		assert(pTmpNet->L[0]->mat==net->L[0]->mat);
		assert(pTmpNet->L[1]->mat==net->L[1]->mat);
		delete pTmpNet;
		delete net;
#endif
	}
	time(&endTime);
	printf("%d seconds elapsed.\n",(int)difftime(endTime,startTime));
}
