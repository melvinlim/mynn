#define LAYERS 2

#define GAMMA (0.1)

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

struct Layer{
	Array *in;
	Matrix *M;
	Matrix *dW;
	Array *out;
	Array *deriv;
};

struct Net{
	Layer **L;
	int size;
};

void updateWeights(Layer *L){
	int i,j;
	Matrix *A=L->M;
//printf("%d %d\n",A->width,A->height);
	Array *delta=L->deriv;
	Array *input=L->in;
	for(i=0;i<A->width;i++){
		for(j=0;j<A->height;j++){
			A->elements[j*A->width+i]-=GAMMA*input->el[i]*delta->el[j];
		}
	}
}

void bpDeltas(Layer *L1,Layer *L2){
	int j,k;
	float sum;
	Array *deriv=L1->deriv;
	Array *delta=L2->deriv;
	Matrix *W=L2->M;
	for(j=0;j<deriv->len;j++){
		sum=0;
		for(k=0;k<delta->len;k++){
			sum+=W->elements[j*W->width+k]*delta->el[k];
		}
		deriv->el[j]=deriv->el[j]*sum;
	}
}

void bpDeltas0(Layer *L,Array *error){
	int j;
	Array *deriv=L->deriv;
	for(j=0;j<deriv->len;j++){
		//storing delta in deriv...  should possibly have dedicated array...
		deriv->el[j]=deriv->el[j]*error->el[j];
	}
}

void nnBackProp(Net *N,Array *error){
	int i;
	bpDeltas0(N->L[LAYERS-1],error);
	for(i=LAYERS-2;i>=0;i--){
		bpDeltas(N->L[i],N->L[i+1]);
	}
	for(i=LAYERS-1;i>=0;i--){
		updateWeights(N->L[i]);
	}
}

//this actually does more than simply multiply.
__global__ void MatMulKernel(const Matrix A, const Array x, Array y, Array deriv){
	int i;
	float Cval=0;
	//int row = blockIdx.y * blockDim.y + threadIdx.y;
  //int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
	for(i=0;i<A.width;i++){
		Cval+=A.elements[row*A.width+i]*x.el[i];
	}
	float tmp=tanhf(Cval);
	y.el[row]=tmp;
	deriv.el[row]=1.0-(tmp*tmp);
}

void MatMul(const Matrix A, const Array x, Array y, Array deriv)
{
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);

		Array d_x;
		d_x.len=x.len;
		d_x.el=x.el;
		size_t xSz=x.len*sizeof(float);
    cudaMalloc(&d_x.el,xSz);
    cudaMemcpy(d_x.el,x.el,xSz,
	    cudaMemcpyHostToDevice);
		
		Array d_y;
		d_y.len=y.len;
		d_y.el=y.el;
		size_t ySz=y.len*sizeof(float);
    cudaMalloc(&d_y.el,ySz);
		
		Array d_deriv;
		d_deriv.len=deriv.len;
		d_deriv.el=deriv.el;
    cudaMalloc(&d_deriv.el,ySz);

    // Invoke kernel
    //dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    //dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    //dim3 dimGrid(dimBlock.x, A.height / dimBlock.y);
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_x, d_y, d_deriv);
    //MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_x, d_y);

    // Read from device memory
    cudaMemcpy(y.el, d_y.el, ySz,
    	cudaMemcpyDeviceToHost);
    
		cudaMemcpy(deriv.el, d_deriv.el, ySz,
    	cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_x.el);
    cudaFree(d_y.el);
    cudaFree(d_deriv.el);
}

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);

    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
    cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}
