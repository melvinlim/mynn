#ifndef _LIBNN
#define _LIBNN

#define EPOCHS 1000
#define LAYERS 2
#define GAMMA (0.1)

#include "types.h"
#include "kernels.cu"

#define TOL (0.001)

class Layer{
public:
	Matrix<double> M;
	Array<double> out;
	Array<double> deriv;
	Array<double> delta;
	Layer(){
	}
	Layer(int m,int n){
		out.resize(m);
		deriv.resize(m);
		delta.resize(m);
		M.resize(m,n);
	}
	~Layer(){
	}
	Array<double> forward(const Array<double> &x){
		int i,j;
		double a,tmp;
		assert(x.n==M.n);
		Array<double> o=this->out;
		Array<double> d=this->deriv;
		forwardGPU(M.m,M.n,this->M.el.data(),x.el.data(),o.el.data(),d.el.data());
		for(j=0;j<M.m;j++){
			a=0;
			for(i=0;i<M.n;i++){
				a+=(M)(j,i)*(x)(i);
			}
			tmp=tanh(a);
			(out)(j)=tmp;
			(deriv)(j)=1.0-(tmp*tmp);
			assert(fabs(out(j)-o[j])<TOL);
			assert(fabs(deriv(j)-d[j])<TOL);
		}
		return(this->out);
	}
	void outputDelta(const Array<double> &error){
		int j;
		assert(error.n==delta.n);
		for(j=0;j<error.n;j++){
			(this->delta)(j)=(this->deriv)(j)*(error)(j);
		}
	}
	void upDelta(const Matrix<double> &W,const Array<double> &delta2){
		int j,k;
		double sum;
		assert(W.n==delta.n);
		assert(W.m==delta2.n);
		Array<double> d=this->delta;
		deltaGPU(W.m,W.n,W.el.data(),d.el.data(),delta2.el.data(),this->deriv.el.data());
		for(j=0;j<W.n;j++){
			sum=0;
			for(k=0;k<W.m;k++){
				sum+=(W)(k,j)*(delta2)(k);
			}
			(this->delta)(j)=(this->deriv)(j)*sum;
			assert(fabs(delta(j)-d[j])<TOL);
		}
	}
	void updateWeights(const Array<double> &input){
		int i,j;
		assert(input.n==M.n);
		assert(delta.n==M.m);
		for(i=0;i<this->M.m;i++){
			for(j=0;j<this->M.n;j++){
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
	vector<Layer> L;
	int n;
	Array<double> error;
	Array<double> answer;
	Net(int n=0){
		this->n=n;
		//this->L.resize(n);
	}
	~Net(){
	}
	void insertLayer(int i,int m,int n){
		//L[i]=Layer(m,n);
		L.push_back(Layer(m,n));
		error.resize(m);
		answer.resize(m);
	}
	void forward(const Array<double> &x){
		//int i;
		L[0].forward(x);
		L[1].forward(L[0].out);
		this->answer=L[1].out;
	}
	void upError(const Array<double> &yTarget){
		int i;
		for(i=0;i<yTarget.n;i++){
//			(this->error)(i)=(this->answer)(i)-(yTarget)(i);
			(this->error)(i)=(L[1].out)(i)-(yTarget)(i);
		}
	}
	void backward(const Array<double> &input){
		//int i;
		L[1].outputDelta(this->error);
		L[0].upDelta(L[1].M,L[1].delta);
/*
		L[LAYERS-1].outputDelta(error);
		for(i=LAYERS-2;i>=0;i--){
			L[i].upDelta(L[i+1].M,L[i+1].delta);
		}
		for(i=LAYERS-1;i>=1;i--){
			L[i].updateWeights(L[i-1].out);
		}
*/
		L[1].updateWeights(L[0].out);
		L[0].updateWeights(input);
	}
	void rand(){
		int i;
		for(i=0;i<this->n;i++){
			this->L[i].rand();
		}
	}
	void print(){
		int i;
		for(i=0;i<this->n;i++){
			L[i].M.print();
		}
	}
	Array<double> train(const Array<double> &x,const Array<double> &y){
		int i;
		L[0].forward(x);
		L[1].forward(L[0].out);
		for(i=0;i<y.n;i++){
			(this->error)(i)=(L[1].out)(i)-(y)(i);
		}
		L[1].outputDelta(this->error);
		L[0].upDelta(L[1].M,L[1].delta);
		L[1].updateWeights(L[0].out);
		L[0].updateWeights(x);
/*
		this->forward(x);
		this->upError(y);
		this->backward(x);
*/
		this->answer=L[1].out;
		return(this->error);
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
