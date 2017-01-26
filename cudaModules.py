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
linearForwardTemplate="""
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
		y[row]=Cval;
		deriv[row]=1.0;
	}
}
"""
deltaTemplate="""
__global__ void deltaKernel(const double *A,double *delta,const double *y,const double *deriv){
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
__global__ void weightKernel(double *A,const double *x,const double *delta){
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
		A[row*%(NCOLS)s+col] += x[col]*delta[row];
	}
}
"""
batchUpdateTemplate="""
__global__ void batchUpdateKernel(double *A,double *dA){
	const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	if((row<%(NROWS)s)&&(col<%(NCOLS)s)){
		A[row*%(NCOLS)s+col] -= %(GAMMA)s*dA[row*%(NCOLS)s+col];
	}
}
"""
batchUpdateADTemplate="""
__global__ void batchUpdateADKernel(double *A,const double *dA,double *grad2,double *theta2){
	const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int ind = row*%(NCOLS)s+col;
	double theta;
	if((row<%(NROWS)s)&&(col<%(NCOLS)s)){
		grad2[ind]=%(GAMMA)s*grad2[ind]+(1-%(GAMMA)s)*(dA[ind]*dA[ind]);
		theta=(-1)*sqrt(theta2[ind]+%(EPSILON)s)/(sqrt(grad2[ind]+%(EPSILON)s))*dA[ind];
		theta2[ind]=%(GAMMA)s*theta2[ind]+(1-%(GAMMA)s)*(theta*theta);
		A[ind] += theta;
	}
}
"""
