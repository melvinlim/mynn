#ifndef _TYPES_H
#define _TYPES_H

#include"defs.h"
#include"matrix.h"
#include"array.h"
class Layer{
public:
	Matrix *M;
	Array *out;
	Array *deriv;
	Array *delta;
	Layer(int n,int m){
		out=new Array(n);
		deriv=new Array(n);
		delta=new Array(n);
		M=new Matrix(n,m);
	}
	~Layer(){
		delete out;
		delete deriv;
		delete delta;
		delete M;
	}
	Array *forward(const Array *x){
		int i,j;
		float a,tmp;
		for(j=0;j<M->n;j++){
			a=0;
			for(i=0;i<M->m;i++){
				a+=(M->atIndex(j,i))*x->item[i];
				//a+=M->item[j*M->m+i]*x->item[i];
				//a+=M->atIndex(j,i)*x->item[i];
			}
			tmp=tanh(a);
			out->item[j]=tmp;
			deriv->item[j]=1.0-(tmp*tmp);
		}
		return(this->out);
	}
	void outputDelta(const Array *error){
		int j;
		for(j=0;j<error->n;j++){
			this->delta->item[j]=this->deriv->item[j]*error->item[j];
			//delta->item[j]=error->item[j];
		}
	}
	void upDelta(const Matrix *W,const Array *delta2){
		int j,k;
		float sum;
		for(j=0;j<this->deriv->n;j++){
			sum=0;
			for(k=0;k<delta2->n;k++){
				sum+=W->item[k*this->deriv->n+j]*delta2->item[k];
				//sum+=(W->atIndex(k,j))*delta2->item[k];
			}
			this->delta->item[j]=this->deriv->item[j]*sum;
		}
	}
	void updateWeights(const Array *input){
		int i,j;
		for(i=0;i<this->M->n;i++){
			for(j=0;j<this->M->m;j++){
				this->M->item[i*this->M->m+j]-=GAMMA*input->item[j]*this->delta->item[i];
			}
		}
	}
	void rand(){
		this->M->rand();
	}
};

class Net{
public:
	Layer **L;
	int n;
	Array *error;
	Array *answer;
	Net(int n=0){
		this->n=n;
		L=(Layer **)malloc(n*sizeof(Layer *));
	}
	~Net(){
		int i;
		for(i=0;i<n;i++){
			free(L[i]);
		}
	}
	void insertLayer(int i,int n,int m){
		L[i]=new Layer(n,m);
		if(i==(this->n-1)){
			error=new Array(n);
			answer=new Array(n);
		}
	}
	void forward(const Array *x){
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
	void backward(const Array *input){
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
	void rand(){
		int i;
		for(i=0;i<this->n;i++){
			this->L[i]->rand();
		}
	}
	void print(){
		int i;
		for(i=0;i<this->n;i++){
			L[i]->M->print();
		}
	}
	Array *train(const Array *x,const Array *y){
		this->forward(x);
		this->upError(y);
		this->backward(x);
		return(this->error);
	}
	void upError(const Array *yTarget){
		int i;
		for(i=0;i<yTarget->n;i++){
			this->error->item[i]=(this->answer->item[i]-yTarget->item[i]);
		}
	}
};

#endif
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
