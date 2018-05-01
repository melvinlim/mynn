#include"net.h"

Net::Net(int n){
	srandom(time(0));
	this->n=n;
	L=new Layer *[n];
	error=new Array(n);
	response=new Array(n);
}
Net::~Net(){
	int i;
	for(i=0;i<n;i++){
		delete L[i];
	}
	delete[] L;
	delete error;
	delete response;
}
void Net::insertLayer(int i,int m,int n){
	L[i]=new Layer(m,n);
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
void Net::randomize(){
	int i;
	for(i=0;i<n;i++){
		L[i]->randomize();
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
#ifdef SOLVEXOR
	status(x,y);
#else
	MNISTStatus(y);
#endif
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
double sumSqError(const Array *array){
	int i;
	int n=array->n;
	double *error=array->item;
	double ret=0;
	for(i=0;i<n;i++){
		ret+=error[i]*error[i];
	}
	return(ret/2.0);
}
int toLabel(double *x){
	int i=0;
	for(i=0;i<10;i++){
		if(*x++>0)	return i;
	}
	return i;
}
void Net::MNISTStatus(const Array *pOut){
	printf("resp:%d targ:%d ssqerr:%f\n",
	toLabel(response->item),
	toLabel(pOut->item),
	sumSqError(error)
	);
}
SingleHidden::SingleHidden(int inputs,int hidden,int outputs):Net(2){
	int L1M=(inputs+1);
	int L1N=(hidden);
	int L2M=(hidden+1);
	int L2N=(outputs);
	insertLayer(0,L1M,L1N);
	insertLayer(1,L2M,L2N);
	randomize();
}
