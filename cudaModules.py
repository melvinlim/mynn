forwardTemplate="""
__global__ void forwardKernel(double *A,double *x,double *y,double *deriv){
	int i;
	double Cval=0;
	//int row = blockIdx.y * blockDim.y + threadIdx.y;
	//int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int row = blockIdx.y * blockDim.y + threadIdx.y;

	if(row<%(NROWS)s){
		for(i=0;i<%(NCOLS)s;i++){
			Cval+=A[row*%(NCOLS)s+i]*x[i];
		}
		double tmp=tanhf(Cval);
		y[row]=tmp;
		deriv[row]=1.0-(tmp*tmp);
	}
}
"""
deltaTemplate="""
__global__ void deltaKernel(double *A,double *x,double *y,double *deriv){
	int i;
	double Cval=0;
	const int col = threadIdx.x;
	for(i=0;i<%(NCOLS)s;i++){
		Cval+=A[i*%(NCOLS)s+col]*y[i];
	}
	x[col]=deriv[col]*Cval;
}
"""
weightTemplate="""
__global__ void weightKernel(double *A,double *x,double *delta){
	const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	if((row<%(NROWS)s)&&(col<%(NCOLS)s)){
		A[row*%(NCOLS)s+col] -= x[col]*delta[row];
	}
}
"""
