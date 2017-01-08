#ifndef TYPES
#define TYPES

#define EPOCHS 1000
#define LAYERS 2
#define GAMMA (0.01)

// Thread block size
//#define BLOCK_SIZE 16
#define BLOCK_SIZE 2

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride; 
    float* elements;
}Matrix;

typedef struct{
	int len;
	float *el;
}Array;

class Layer{
public:
	Array *in;
	Matrix *M;
	Matrix *dW;
	Array *out;
	Array *deriv;
	Array *delta;
	Layer(int n,int m){
/*
		out=(Array *)malloc(sizeof(Array));
		deriv=(Array *)malloc(sizeof(Array));
		delta=(Array *)malloc(sizeof(Array));
		in->el=(float *)malloc(mDim[i]*sizeof(float));
		out->el=(float *)malloc(nDim[i]*sizeof(float));
		deriv->el=(float *)malloc(nDim[i]*sizeof(float));
		delta->el=(float *)malloc(nDim[i]*sizeof(float));
		M=(Matrix *)malloc(sizeof(Matrix));
		M->elements=(float *)malloc(nDim[i]*mDim[i]*sizeof(float));
		dW=(Matrix *)malloc(sizeof(Matrix));
		dW->elements=(float *)malloc(nDim[i]*mDim[i]*sizeof(float));
*/
	}
	~Layer(){
	}
};

class Net{
public:
	Layer **L;
	int n;
	Net(int n=0){
		int i;
		this->n=n;
		L=(Layer **)malloc(n*sizeof(Layer *));
		//L=new Layer(5,5);
	}
	~Net(){
		int i;
		for(i=0;i<n;i++){
			//free(L[i]);
		}
	}
	void insertLayer(int i,int n,int m){
		L[i]=new Layer(n,m);
	}
};

#endif
