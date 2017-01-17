#ifndef _KERNELS
#define _KERNELS

//#include "types.h"

#define BLOCK_SIZE 32
#define GAMMA (0.1)

__global__ void forwardKernel(const int m,const int n,const double *A, const double *x, double *y, double *deriv){
	int j;
	double Cval=0;
	//int row = blockIdx.y * blockDim.y + threadIdx.y;
  //int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row<m){
		for(j=0;j<n;j++){
			Cval+=A[row*n+j]*x[j];
		}
		double tmp=tanhf(Cval);
		y[row]=tmp;
		deriv[row]=1.0-(tmp*tmp);
	}
}
void forwardGPU(const int m,const int n,const double *A, const double *x, double *y, double *deriv){
    size_t size = m * n * sizeof(double);
		double *d_A=(double *)malloc(size);
    cudaMalloc(&d_A,size);
    cudaMemcpy(d_A,A,size,
               cudaMemcpyHostToDevice);

		size_t xSz=n*sizeof(double);
		double *d_x=(double *)malloc(xSz);
    cudaMalloc(&d_x,xSz);
    cudaMemcpy(d_x,x,xSz,
	    cudaMemcpyHostToDevice);
		
		size_t ySz=m*sizeof(double);
		double *d_y=(double *)malloc(ySz);
    cudaMalloc(&d_y,ySz);
		
		double *d_deriv=(double *)malloc(ySz);
    cudaMalloc(&d_deriv,ySz);

    // Invoke kernel
    //dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    //dim3 dimGrid(B.m / dimBlock.x, A.n / dimBlock.y);
    //dim3 dimGrid(dimBlock.x, A.n / dimBlock.y);

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((int)ceil((float)m/(float)dimBlock.x));
    //dim3 dimBlock(m);
    //dim3 dimGrid(1);
    forwardKernel<<<dimGrid, dimBlock>>>(m,n,d_A, d_x, d_y, d_deriv);

    // Read from device memory
    cudaMemcpy(y,d_y,ySz,
    	cudaMemcpyDeviceToHost);
    
		cudaMemcpy(deriv,d_deriv,ySz,
    	cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_deriv);
}

__global__ void deltaKernel(const int m,const int n,const double *A,double *delta,const double *y,const double *deriv){
	int i;
	double Cval=0;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(col<n){
		for(i=0;i<m;i++){
			Cval+=A[i*n+col]*y[i];
		}
		delta[col]=deriv[col]*Cval;
	}
}
void deltaGPU(const int m,const int n,const double *A,double *delta,const double *y,const double *deriv){
    size_t size = m * n * sizeof(double);
		double *d_A=(double *)malloc(size);
    cudaMalloc(&d_A,size);
    cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);

		size_t xSz=n*sizeof(double);
		double *d_delta=(double *)malloc(xSz);
    cudaMalloc(&d_delta,xSz);
//    cudaMemcpy(d_delta,delta,xSz,cudaMemcpyHostToDevice);
		
		size_t ySz=m*sizeof(double);
		double *d_y=(double *)malloc(ySz);
    cudaMalloc(&d_y,ySz);
    cudaMemcpy(d_y,y,ySz,cudaMemcpyHostToDevice);
		
		double *d_deriv=(double *)malloc(xSz);
    cudaMalloc(&d_deriv,xSz);
    cudaMemcpy(d_deriv,deriv,xSz,cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((int)ceil((float)n/(float)dimBlock.x));
    //dim3 dimBlock(m);
    //dim3 dimGrid(1);
    deltaKernel<<<dimGrid,dimBlock>>>(m,n,d_A,d_delta,d_y,d_deriv);

    cudaMemcpy(delta,d_delta,xSz,cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_delta);
    cudaFree(d_y);
    cudaFree(d_deriv);
}
__global__ void weightKernel(const int m,const int n,double *A,double *x,double *delta){
	const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	if((row<m)&&(col<n)){
		A[row*n+col] -= GAMMA*x[col]*delta[row];
	}
}
__global__ void batchAccumKernel(const int m,const int n,double *A,double *x,double *delta){
	const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	if((row<m)&&(col<n)){
		A[row*n+col] += GAMMA*x[col]*delta[row];
	}
}
__global__ void batchUpdateKernel(const int m,const int n,double *A,double *dA){
	const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	if((row<m)&&(col<n)){
		A[row*n+col] -= dA[row*n+col];
	}
}
__global__ void batchUpdateDMKernel(const int m,const int n,double *A,double *dA){
	const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	if((row<m)&&(col<n)){
		double x;
		x=dA[row*n+col];
		if(x>1)
			A[row*n+col] -= 1;
		else if(x<(-1))
			A[row*n+col] += 1;
		else
			A[row*n+col] -= x;
	}
}

#endif
