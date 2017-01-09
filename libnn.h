#ifndef _LIBNN
#define _LIBNN

#define EPOCHS 1000
#define LAYERS 2
#define GAMMA (0.01)

// Thread block size
//#define BLOCK_SIZE 16
#define BLOCK_SIZE 2

#define RANDSCALING 10	//scale random weights to be from -0.1 to +0.1

#include "types.h"

class Layer{
public:
	Matrix<double> M;
	Array<double> out;
	Array<double> deriv;
	Array<double> delta;
	Layer(int n,int m){
		out.resize(n);
		deriv.resize(n);
		delta.resize(n);
		M.resize(n,m);
	}
	~Layer(){
	}
	Array<double> forward(const Array<double> &x){
		int i,j;
		double a,tmp;
		for(j=0;j<M.n;j++){
			a=0;
			for(i=0;i<M.m;i++){
				a+=(M)(j,i)*(x)(i);
				//a+=M->el[j*M->m+i]*x->el[i];
				//a+=M->e(j,i)*x->el[i];
			}
			tmp=tanh(a);
			(out)(j)=tmp;
			(deriv)(j)=1.0-(tmp*tmp);
		}
		return(this->out);
	}
	void outputDelta(const Array<double> &error){
		int j;
		for(j=0;j<error.n;j++){
			(this->delta)(j)=(this->deriv)(j)*(error)(j);
			//delta->el[j]=error->el[j];
		}
	}
	void upDelta(const Matrix<double> &W,const Array<double> &delta2){
		int j,k;
		double sum;
		//for(j=0;j<this->deriv->n;j++){
		for(j=0;j<W.m;j++){
			sum=0;
			//for(k=0;k<delta2->n;k++){
			for(k=0;k<W.n;k++){
				sum+=(W)(k,j)*(delta2)(k);
				//sum+=W->el[k*this->deriv->n+j]*delta2->el[k];
				//sum+=(*(W->e(k,j)))*delta2->el[k];
			}
			(this->delta)(j)=(this->deriv)(j)*sum;
		}
	}
	void updateWeights(const Array<double> &input){
		int i,j;
		for(i=0;i<this->M.n;i++){
			for(j=0;j<this->M.m;j++){
				(this->M)(i,j)-=GAMMA*(input)(j)*(this->delta)(i);
			}
		}
	}
	void rand(){
		this->M.rand();
	}
};

class Net{
public:
	Layer **L;
	int n;
	Array<double> error;
	Array<double> answer;
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
		error.resize(n);
		answer.resize(n);
	}
	void forward(const Array<double> &x){
		//int i;
		L[0]->forward(x);
		L[1]->forward(L[0]->out);
/*
		Array<double> *t=x;
		for(i=0;i<this->n;i++){
			t=L[i]->forward(t);
	//		MatMul(*N->L[i]->M,*N->L[i]->in,*N->L[i]->out,*N->L[i]->deriv);
		}
*/
		this->answer=L[1]->out;
	}
	void backward(const Array<double> &input){
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
			L[i]->M.print();
		}
	}
	Array<double> train(const Array<double> &x,const Array<double> &y){
		this->forward(x);
		this->upError(y);
		this->backward(x);
		return(this->error);
	}
	void upError(const Array<double> &yTarget){
		int i;
		for(i=0;i<yTarget.n;i++){
//			(this->error)(i)=(this->answer)(i)-(yTarget)(i);
			(this->error)(i)=(L[1]->out)(i)-(yTarget)(i);
		}
	}
};

#endif
/*
double nnTotalError(const Array<double> *y0,const Array<double> *y){
	int i;
	int n=y0->n;
	double ret=0;
	for(i=0;i<n;i++){
		ret+=fabs(y0->el[i]-y->el[i]);
		ret*=ret;
	}
	return(ret/2.0);
}
*/
