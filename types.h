#ifndef _TYPES
#define _TYPES

#define EPOCHS 1000
#define LAYERS 2
#define GAMMA (0.01)

// Thread block size
//#define BLOCK_SIZE 16
#define BLOCK_SIZE 2

#define RANDSCALING 10	//scale random weights to be from -0.1 to +0.1

class Matrix{
public:
	int n;
	int m;
	float *el;
	Matrix(int n,int m){
		int i;
		this->n=n;
		this->m=m;
		el=new float[n*m];
		for(i=0;i<n*m;i++){
			el[i]=0;
		}
	}
	~Matrix(){
	}
	float *e(int i,int j){
		if(((i+1)*(j+1))>(this->n*this->m)){
			printf("dimension error\n");
			exit(1);
		}
		return(this->el+(i*this->m+j));
	}
	void rand(){
		int i,j;
		for(i=0;i<this->n;i++){
			for(j=0;j<this->m;j++){
				*this->e(i,j)=
				(random()-(RAND_MAX/2))*2.0/((float)RAND_MAX)/((float)RANDSCALING);
			}
		}
	}
	void print(){
		int i,j;
		for(i=0;i<this->n;i++){
			for(j=0;j<this->m;j++){
				printf("[%i,%i]%.09f ",i,j,*this->e(i,j));
			}
			printf("\n");
		}
		printf("\n");
	}
};

class Array{
public:
	int n;
	float *el;
	Array(int n){
		int i;
		this->n=n;
		el=new float[n];
		for(i=0;i<n;i++){
			el[i]=0;
		}
	}
	Array(const float *x,const int n){
		int i;
		this->n=n;
		this->el=new float[n];
		if(x){
			for(i=0;i<n;i++){
				this->el[i]=x[i];
			}
//			memcpy(p->el,x,n*sizeof(float));
		}
	}
	~Array(){
	}
	void print(){
		int i;
		float *x;
		x=this->el;
		for(i=0;i<this->n;i++){
			printf("[%i]%.02f\t",i,*x++);
		}
		printf("\n");
	}
	void rand(){
		int i;
		for(i=0;i<this->n;i++){
			this->el[i]=
			(random()-(RAND_MAX/2))*2.0/((float)RAND_MAX)/((float)RANDSCALING);
		}
	}
};

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
	}
	Array *forward(const Array *x){
		int i,j;
		float a,tmp;
		for(j=0;j<M->n;j++){
			a=0;
			for(i=0;i<M->m;i++){
				a+=(*(M->e(j,i)))*x->el[i];
				//a+=M->el[j*M->m+i]*x->el[i];
				//a+=M->e(j,i)*x->el[i];
			}
			tmp=tanh(a);
			out->el[j]=tmp;
			deriv->el[j]=1.0-(tmp*tmp);
		}
		return(this->out);
	}
	void outputDelta(const Array *error){
		int j;
		for(j=0;j<error->n;j++){
			this->delta->el[j]=this->deriv->el[j]*error->el[j];
			//delta->el[j]=error->el[j];
		}
	}
	void upDelta(const Matrix *W,const Array *delta2){
		int j,k;
		float sum;
		for(j=0;j<this->deriv->n;j++){
			sum=0;
			for(k=0;k<delta2->n;k++){
				sum+=W->el[k*this->deriv->n+j]*delta2->el[k];
				//sum+=(*(W->e(k,j)))*delta2->el[k];
			}
			this->delta->el[j]=this->deriv->el[j]*sum;
		}
	}
	void updateWeights(const Array *input){
		int i,j;
		for(i=0;i<this->M->n;i++){
			for(j=0;j<this->M->m;j++){
				this->M->el[i*this->M->m+j]-=GAMMA*input->el[j]*this->delta->el[i];
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
			this->error->el[i]=(this->answer->el[i]-yTarget->el[i]);
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
		ret+=fabs(y0->el[i]-y->el[i]);
		ret*=ret;
	}
	return(ret/2.0);
}
*/
