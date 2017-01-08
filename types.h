#ifndef TYPES
#define TYPES

#define EPOCHS 1000
#define LAYERS 2
#define GAMMA (0.01)

// Thread block size
//#define BLOCK_SIZE 16
#define BLOCK_SIZE 2

class Matrix{
public:
	int n;
	int m;
	float *el;
	Matrix(int n,int m){
		this->n=n;
		this->m=m;
		el=new float(n*m);
	}
	~Matrix(){
	}
};

class Array{
public:
	int len;
	float *el;
	Array(int n){
		len=n;
		el=new float(n);
	}
	Array(const float *x,int n){
		int i;
		this->len=n;
		this->el=new float(n);
		if(x){
			for(i=0;i<n;i++){
				this->el[i]=x[i];
			}
//			memcpy(p->el,x,n*sizeof(float));
		}
	}
	~Array(){
	}
};

class Layer{
public:
	Matrix *M;
	Matrix *dW;
	Array *out;
	Array *deriv;
	Array *delta;
	Layer(int n,int m){
		out=new Array(n);
		deriv=new Array(n);
		delta=new Array(n);
		M=new Matrix(n,m);
		dW=new Matrix(n,m);
	}
	~Layer(){
	}
};

class Net{
public:
	Layer **L;
	int n;
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
	}
	void input(Array *in){
	}
};

#endif
