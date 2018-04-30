#include"net.h"

Net::Net(int n){
	this->n=n;
	L=(Layer **)malloc(n*sizeof(Layer *));
}
Net::~Net(){
	int i;
	for(i=0;i<n;i++){
		free(L[i]);
	}
}
void Net::insertLayer(int i,int n,int m){
	L[i]=new Layer(n,m);
	if(i==(this->n-1)){
		error=new Array(n);
		answer=new Array(n);
	}
}
void Net::forward(const Array *x){
	//int i;
	L[0]->forward(x);
	L[1]->forward(L[0]->out);
/*
	Array *t=x;
	for(i=0;i<this->n;i++){
		t=L[i]->forward(t);
//		MatMul(*N->L[i]->M,*N->L[i]->in,*N->L[i]->out,*N->L[i]->deriv);
	}
*/
	this->answer=L[1]->out;
}
void Net::backward(const Array *input){
	//int i;
	L[1]->outputDelta(this->error);
	L[0]->upDelta(L[1]->M,L[1]->delta);
/*
	L[LAYERS-1]->outputDelta(error);
	for(i=LAYERS-2;i>=0;i--){
		L[i]->upDelta(L[i+1]->M,L[i+1]->delta);
	}
	for(i=LAYERS-1;i>=1;i--){
		L[i]->updateWeights(L[i-1]->out);
	}
*/
	L[1]->updateWeights(L[0]->out);
	L[0]->updateWeights(input);
}
void Net::rand(){
	int i;
	for(i=0;i<this->n;i++){
		this->L[i]->rand();
	}
}
void Net::print(){
	int i;
	for(i=0;i<this->n;i++){
		L[i]->M->print();
	}
}
Array *Net::train(const Array *x,const Array *y){
	forward(x);
	updateError(y);
	backward(x);
	status(x,y);
	return(this->error);
}
void Net::updateError(const Array *yTarget){
	int i;
	for(i=0;i<yTarget->n;i++){
		this->error->item[i]=(this->answer->item[i]-yTarget->item[i]);
	}
}
void Net::status(const Array *pIn,const Array *pOut){
	printf("in:[%.0f,%.0f] ans:[%f,%f] targ:[%.0f,%.0f] err:[%f,%f]\n",
	pIn->item[0],pIn->item[1],
	answer->item[0],answer->item[1],
	pOut->item[0],pOut->item[1],
	error->item[0],error->item[1]
	);
}

/*
float nnTotalError(const Array *y0,const Array *y){
int i;
int n=y0->n;
float ret=0;
for(i=0;i<n;i++){
	ret+=fabs(y0->item[i]-y->item[i]);
	ret*=ret;
}
return(ret/2.0);
}
*/
