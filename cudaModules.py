forwardTemplate="""
__global__ void forwardKernel(const double *A,const double *x,double *y,double *deriv){
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
__global__ void deltaKernel(double *A,double *delta,double *y,double *deriv){
	int i;
	double Cval=0;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(col<%(NCOLS)s){
		for(i=0;i<%(NROWS)s;i++){
			Cval+=A[i*%(NCOLS)s+col]*y[i];
		}
		delta[col]=deriv[col]*Cval;
	}
}
"""
weightTemplate="""
__global__ void weightKernel(double *A,double *x,double *delta){
	const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	if((row<%(NROWS)s)&&(col<%(NCOLS)s)){
		A[row*%(NCOLS)s+col] -= %(GAMMA)s*x[col]*delta[row];
	}
}
"""
batchAccumTemplate="""
__global__ void batchAccumKernel(double *A,double *x,double *delta){
	const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	if((row<%(NROWS)s)&&(col<%(NCOLS)s)){
		A[row*%(NCOLS)s+col] += %(GAMMA)s*x[col]*delta[row];
	}
}
"""
batchUpdateTemplate="""
__global__ void batchUpdateKernel(double *A,double *dA){
	const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	if((row<%(NROWS)s)&&(col<%(NCOLS)s)){
		A[row*%(NCOLS)s+col] -= dA[row*%(NCOLS)s+col];
	}
}
"""
batchUpdateDMTemplate="""
__global__ void batchUpdateDMKernel(double *A,double *dA){
	const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	if((row<%(NROWS)s)&&(col<%(NCOLS)s)){
		double x;
		x=dA[row*%(NCOLS)s+col];
		if(x>1)
			A[row*%(NCOLS)s+col] -= 1;
		else if(x<(-1))
			A[row*%(NCOLS)s+col] += 1;
		else
			A[row*%(NCOLS)s+col] -= x;
	}
}
"""
