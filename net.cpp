#include"net.h"

Net::Net(int n){
	srand(time(0));
	this->n=n;
	L=new Layer *[n];
}
Net::~Net(){
	int i;
	for(i=0;i<n;i++){
		delete L[i];
	}
	delete[] L;
}
void Net::insertLayer(int i,int m,int n){
	L[i]=new Layer(m,n);
	if(i==(this->n-1)){
		error=new Array(n);
		response=new Array(n);
	}
}
void Net::forward(const Array *x){
	L[0]->forward(x);
	L[1]->forward(L[0]->out);
	response=L[1]->out;
}
void Net::backward(const Array *input){
	L[1]->outputDelta(error);
	L[0]->hiddenDelta(L[1]->mat,L[1]->delta);
	L[1]->saveErrors(L[0]->out);
	L[0]->saveErrors(input);
/*
	L[1]->directUpdateWeights(L[0]->out);
	L[0]->directUpdateWeights(input);
*/
}
void Net::rand(){
	int i;
	for(i=0;i<n;i++){
		L[i]->rand();
	}
}
void Net::print(){
	int i;
	for(i=0;i<n;i++){
		L[i]->mat->print();
	}
}
void Net::updateWeights(){
	L[1]->updateWeights();
	L[0]->updateWeights();
}
Array *Net::train(const Array *x,const Array *y){
	forward(x);
	updateError(y);
	backward(x);
	status(x,y);
	return(error);
}
void Net::updateError(const Array *yTarget){
	int i;
	for(i=0;i<yTarget->n;i++){
		error->item[i]=yTarget->item[i]-response->item[i];
	}
}
void Net::status(const Array *pIn,const Array *pOut){
	printf("in:[%.0f,%.0f] ans:[%f,%f] targ:[%.0f,%.0f] err:[%f,%f]\n",
	pIn->item[0],pIn->item[1],
	response->item[0],response->item[1],
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
