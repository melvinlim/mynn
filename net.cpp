#include"net.h"

Net::Net(int n,int outputs):
error(outputs)
{
	srandom(time(0));
	this->n=n;
	L=new Layer *[n];
	response=0;
}
Net::~Net(){
	int i;
	if(L){
		for(i=0;i<n;i++){
			delete L[i];
		}
	}
	delete[] L;
}
void Net::insertLayer(int i,Matrix<double> &mat,double gamma){
	L[i]=new Layer(mat,gamma);
}
void Net::insertLayer(int i,int m,int n,double gamma){
	L[i]=new Layer(m,n,gamma);
}
void Net::forward(const Array<double> *x){
	L[0]->forward(*x);
	L[1]->forward(L[0]->out);
}
inline void Net::backward(){
	L[1]->outputDelta(error);
	L[0]->hiddenDelta(L[1]->mat,L[1]->delta);
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
		L[i]->mat.print();
	}
}
inline void Net::updateBatchCorrections(const Array<double> *input){
	L[1]->saveErrors(L[0]->out);
	L[0]->saveErrors(*input);
}
inline void Net::directUpdateWeights(const Array<double> *input){
	L[1]->directUpdateWeights(L[0]->out);
	L[0]->directUpdateWeights(*input);
}
void Net::updateWeights(){
	L[1]->updateWeights();
	L[0]->updateWeights();
}
Array<double> &Net::trainOnce(const Array<double> *x,const Array<double> *y){
	forward(x);
	updateError(y);
	backward();
	directUpdateWeights(x);
	return(error);
}
Array<double> &Net::trainBatch(const Array<double> *x,const Array<double> *y){
	forward(x);
	updateError(y);
	backward();
	updateBatchCorrections(x);
	return(error);
}
void Net::updateError(const Array<double> *yTarget){
	int i;
	for(i=0;i<yTarget->nElements;i++){
		error.item[i]=yTarget->item[i]-response->item[i];
	}
}
SingleHidden::SingleHidden(int inputs,int hidden,int outputs,double gamma):Net(2,outputs){
	int L1M=(inputs+1);
	int L1N=(hidden);
	int L2M=(hidden+1);
	int L2N=(outputs);
	insertLayer(0,L1M,L1N,gamma);
	insertLayer(1,L2M,L2N,gamma);
	response=&L[n-1]->out;
	randomize();
}
SingleHiddenLinear::SingleHiddenLinear(int inputs,int hidden,int outputs,double gamma):Net(2,outputs){
	int L1M=(inputs+1);
	int L1N=(hidden);
	int L2M=(hidden+1);
	int L2N=(outputs);
	insertLayer(0,L1M,L1N,gamma);
	insertLayer(1,L2M,L2N,gamma);
	response=&L[n-1]->out;
	randomize();
}
