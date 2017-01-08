void updateWeights(Layer *L){
	int i,j;
	Matrix *A=L->M;
//printf("%d %d\n",A->m,A->n);
	Array *delta=L->delta;
	//Array *input=L->in;
	for(j=0;j<A->n;j++){
		for(i=0;i<A->m;i++){
			//A->el[j*A->m+i]-=GAMMA*input->el[i]*delta->el[j];
		}
	}
}

void bpDeltas(Layer *L1,Layer *L2){
	int j,k;
	float sum;
	Array *deriv=L1->deriv;
	Array *delta1=L1->delta;
	Array *delta2=L2->delta;
	Matrix *W=L2->M;
	for(j=0;j<deriv->len;j++){
		sum=0;
		for(k=0;k<delta2->len;k++){
			sum+=W->el[j*W->m+k]*delta2->el[k];
		}
		delta1->el[j]=deriv->el[j]*sum;
	}
}

void bpDeltas0(Layer *L,const Array *error){
	int j;
	Array *deriv=L->deriv;
	Array *delta=L->delta;
	for(j=0;j<deriv->len;j++){
		delta->el[j]=deriv->el[j]*error->el[j];
		//delta->el[j]=error->el[j];
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

void layerForward(const Matrix *M,const Array *in,Array *out,Array *deriv){
	int i,j;
	float a,tmp;
	for(j=0;j<M->n;j++){
		a=0;
		for(i=0;i<M->m;i++){
			a+=M->el[j*M->m+i]*in->el[i];
		}
		tmp=tanh(a);
		out->el[j]=tmp;
		deriv->el[j]=1.0-tmp*tmp;
	}
}

Array *nnForward(Net *N){
	int i;
	Matrix *M;
	Array *in;
	Array *out;
	Array *deriv;
	for(i=0;i<LAYERS;i++){
		M=N->L[i]->M;
		//in=N->L[i]->in;
		out=N->L[i]->out;
		deriv=N->L[i]->deriv;
//PRINTARRAY(in);
		layerForward(M,in,out,deriv);
//PRINTARRAY(out);
//		MatMul(*N->L[i]->M,*N->L[i]->in,*N->L[i]->out,*N->L[i]->deriv);
	}
	return out;
}

//this actually does more than simply multiply.
__global__ void MatMulKernel(const Matrix A, const Array x, Array y, Array deriv){
	int i;
	float Cval=0;
	//int row = blockIdx.y * blockDim.y + threadIdx.y;
  //int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
	for(i=0;i<A.m;i++){
		Cval+=A.el[row*A.m+i]*x.el[i];
	}
	float tmp=tanhf(Cval);
	y.el[row]=tmp;
	deriv.el[row]=1.0-(tmp*tmp);
}
/*
void MatMul(const Matrix A, const Array x, Array y, Array deriv)
{
    Matrix d_A;
    d_A.m = d_A.m = A.m; d_A.n = A.n;
    size_t size = A.m * A.n * sizeof(float);
    cudaMalloc(&d_A.el, size);
    cudaMemcpy(d_A.el, A.el, size,
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
    //dim3 dimGrid(B.m / dimBlock.x, A.n / dimBlock.y);
    //dim3 dimGrid(dimBlock.x, A.n / dimBlock.y);
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(A.n / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_x, d_y, d_deriv);
    //MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_x, d_y);

    // Read from device memory
    cudaMemcpy(y.el, d_y.el, ySz,
    	cudaMemcpyDeviceToHost);
    
		cudaMemcpy(deriv.el, d_deriv.el, ySz,
    	cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.el);
    cudaFree(d_x.el);
    cudaFree(d_y.el);
    cudaFree(d_deriv.el);
}
*/
// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.el[row * A.m + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.el[row * A.m + col] = value;
}
/*
// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
{
    Matrix Asub;
    Asub.m    = BLOCK_SIZE;
    Asub.n   = BLOCK_SIZE;
    Asub.m   = A.m;
    Asub.el = &A.el[A.m * BLOCK_SIZE * row
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
    d_A.m = d_A.m = A.m; d_A.n = A.n;
    size_t size = A.m * A.n * sizeof(float);
    cudaMalloc(&d_A.el, size);
    cudaMemcpy(d_A.el, A.el, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.m = d_B.m = B.m; d_B.n = B.n;
    size = B.m * B.n * sizeof(float);

    cudaMalloc(&d_B.el, size);
    cudaMemcpy(d_B.el, B.el, size,
    cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.m = d_C.m = C.m; d_C.n = C.n;
    size = C.m * C.n * sizeof(float);
    cudaMalloc(&d_C.el, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.m / dimBlock.x, A.n / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.el, d_C.el, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.el);
    cudaFree(d_B.el);
    cudaFree(d_C.el);
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
    for (int m = 0; m < (A.m / BLOCK_SIZE); ++m) {

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
*/
