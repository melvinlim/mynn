#include"layer.h"
Layer::Layer(int m,int n){
	out=new Array(n);
	deriv=new Array(n);
	delta=new Array(n);
	M=new Matrix(m,n);
}
Layer::~Layer(){
	delete out;
	delete deriv;
	delete delta;
	delete M;
}
Array *Layer::forward(const Array *x){
	int i,j;
	float a,tmp;
	for(j=0;j<M->n;j++){
		a=0;
		for(i=0;i<M->m;i++){
			a+=(M->atIndex(i,j))*x->item[i];
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
		this->delta->item[j]=this->deriv->item[j]*error->item[j];
		//delta->item[j]=error->item[j];
	}
}
void Layer::upDelta(const Matrix *W,const Array *delta2){
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
void Layer::updateWeights(const Array *input){
	int i,j;
	for(i=0;i<this->M->n;i++){
		for(j=0;j<this->M->m;j++){
			this->M->item[i*this->M->m+j]-=GAMMA*input->item[j]*this->delta->item[i];
		}
	}
}
void Layer::rand(){
	M->rand();
}
