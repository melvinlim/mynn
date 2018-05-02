#include"layer.h"
Layer::Layer(Matrix<double> &mat,double gamma):
mat(mat),
dw(mat.nRows,mat.nCols)
{
	int m=mat.nRows;
	int n=mat.nCols;
	nRows=m;
	nCols=n;
	this->gamma=gamma;
	out=new Array(n);
	deriv=new Array(n);
	delta=new Array(n);
}
Layer::Layer(int m,int n,double gamma):
mat(m,n),
dw(m,n)
{
	nRows=m;
	nCols=n;
	this->gamma=gamma;
	out=new Array(n);
	deriv=new Array(n);
	delta=new Array(n);
}
Layer::~Layer(){
	delete out;
	delete deriv;
	delete delta;
}
Array *Layer::forward(const Array *x){
	int i,j;
	double a,tmp;
	for(j=0;j<nCols;j++){
		a=0;
		for(i=0;i<x->n;i++){
			a+=(mat.atIndex(i,j))*x->item[i];
		}
		a+=(mat.atIndex(i,j));
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
void Layer::hiddenDelta(const Matrix<double> &W,const Array *delta2){
	int j,k;
	double sum;
	//assert(W->m==delta->n);
	//assert(W->n==delta2->n);
	for(j=0;j<this->deriv->n;j++){
		sum=0;
		for(k=0;k<delta2->n;k++){
			sum+=(W.atIndex(j,k))*delta2->item[k];
		}
		this->delta->item[j]=this->deriv->item[j]*sum;
	}
}
void Layer::updateWeights(){
	int i,j;
	for(j=0;j<nCols;j++){
		for(i=0;i<nRows;i++){
			mat.item[i*nCols+j]+=dw.item[i*nCols+j];
			dw.item[i*nCols+j]=0;
		}
	}
}
void Layer::saveErrors(const Array *input){
	int i,j;
	for(j=0;j<nCols;j++){
		for(i=0;i<input->n;i++){
			dw.item[i*nCols+j]+=gamma*input->item[i]*this->delta->item[j];
		}
		dw.item[i*nCols+j]+=gamma*this->delta->item[j];
	}
}
void Layer::directUpdateWeights(const Array *input){
	int i,j;
	for(j=0;j<nCols;j++){
		for(i=0;i<input->n;i++){
			mat.item[i*nCols+j]+=gamma*input->item[i]*this->delta->item[j];
		}
		mat.item[i*nCols+j]+=gamma*this->delta->item[j];
	}
}
void Layer::randomize(){
	mat.randomize(RANDSCALING);
}
