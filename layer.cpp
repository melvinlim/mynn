#include"layer.h"
Layer::Layer(int m,int n){
	out=new Array(n);
	deriv=new Array(n);
	delta=new Array(n);
	mat=new Matrix(m,n);
	dw=new Matrix(m,n);
}
Layer::~Layer(){
	delete out;
	delete deriv;
	delete delta;
	delete mat;
}
Array *Layer::forward(const Array *x){
	int i,j;
	double a,tmp;
	for(j=0;j<mat->n;j++){
		a=0;
		for(i=0;i<mat->m;i++){
			a+=(mat->atIndex(i,j))*x->item[i];
		}
		tmp=tanh(a);
		out->item[j]=tmp;
		deriv->item[j]=1.0-(tmp*tmp);
	}
	return(this->out);
}
void Layer::outputDelta(const Array *error){
	int j;
	for(j=0;j<error->n;j++){
		delta->item[j]=deriv->item[j]*error->item[j];
	}
}
void Layer::hiddenDelta(const Matrix *W,const Array *delta2){
	int j,k;
	double sum;
	assert(W->m==delta->n);
	assert(W->n==delta2->n);
	for(j=0;j<this->deriv->n;j++){
		sum=0;
		for(k=0;k<delta2->n;k++){
			sum+=(W->atIndex(j,k))*delta2->item[k];
		}
		this->delta->item[j]=this->deriv->item[j]*sum;
	}
}
void Layer::updateWeights(){
	int i,j;
	for(i=0;i<mat->m;i++){
		for(j=0;j<mat->n;j++){
			mat->item[i*mat->n+j]+=dw->item[i*mat->n+j];
			dw->item[i*mat->n+j]=0;
		}
	}
}
void Layer::saveErrors(const Array *input){
	int i,j;
	for(i=0;i<mat->m;i++){
		for(j=0;j<mat->n;j++){
			dw->item[i*mat->n+j]+=GAMMA*input->item[i]*this->delta->item[j];
		}
	}
}
void Layer::directUpdateWeights(const Array *input){
	int i,j;
	for(i=0;i<mat->m;i++){
		for(j=0;j<mat->n;j++){
			mat->item[i*mat->n+j]+=GAMMA*input->item[i]*this->delta->item[j];
/*
			mat->item[i*mat->n+j]+=0.1*input->item[i]*this->delta->item[j];
			mat->item[i*mat->n+j]+=0.001*dw->atIndex(i,j);
			dw->item[i*mat->n+j]=mat->atIndex(i,j);
*/
		}
	}
}
void Layer::rand(){
	mat->rand();
}
