#ifndef _TYPES
#define _TYPES

#define RANDSCALING (10.0)	//scale random weights to be from -0.1 to +0.1

template<typename T>
class Matrix{
public:
	int m;
	int n;
	std::vector<T> el;
	Matrix(){
		this->m=0;
		this->n=0;
	}
	Matrix(int m,int n){
		int i;
		this->m=m;
		this->n=n;
		el.resize(m*n);
		for(i=0;i<m*n;i++){
			el[i]=0;
		}
	}
	~Matrix(){
	}
	void resize(int m,int n){
		this->m=m;
		this->n=n;
		el.resize(m*n);
	}
	T &operator()(unsigned int i,unsigned int j){
		assert((i<this->m)&&(j<this->n));
		return(el[(i*(this->n))+j]);
	}
	const T &operator()(unsigned int i,unsigned int j) const{
		assert((i<this->m)&&(j<this->n));
		return(el[(i*(this->n))+j]);
	}
	Matrix<T> &operator=(const Matrix<T> &rhs){
		int i,j;
		this->resize(rhs.m,rhs.n);
		for(i=0;i<rhs.m;i++){
			for(j=0;j<rhs.n;j++){
				this->el(i,j)=rhs.el(i,j);
			}
		}
		return *this;
	}
	void rand(){
		int i,j;
		for(i=0;i<this->m;i++){
			for(j=0;j<this->n;j++){
				(*this)(i,j)=
				(random()-(RAND_MAX/2))*2.0/((double)RAND_MAX)/((double)RANDSCALING);
			}
		}
	}
	void print() const{
		int i,j;
		for(i=0;i<this->m;i++){
			for(j=0;j<this->n;j++){
				printf("[%i,%i]%.09f ",i,j,(*this)(i,j));
			}
			printf("\n");
		}
		printf("\n");
	}
};

template<typename T>
class Array{
public:
	int n;
	std::vector<T> el;
	void resize(int n){
		this->n=n;
		el.resize(n);
	}
	Array(){
		this->n=0;
	}
	Array(int n){
		int i;
		this->n=n;
		el.resize(n);
		for(i=0;i<n;i++){
			el[i]=0;
		}
	}
	Array(double *x,int n){
		int i;
		this->n=n;
		this->el.resize(n);
		assert(x);
		for(i=0;i<n;i++){
			this->el[i]=x[i];
		}
//			memcpy(p->el,x,n*sizeof(double));
	}
	~Array(){
	}
	T &operator()(unsigned int i){
		assert(i<this->n);
		return(el[i]);
	}
	const T &operator()(unsigned int i) const{
		assert(i<this->n);
		return(el[i]);
	}
	T &operator[](unsigned int i){
		assert(i<this->n);
		return(el[i]);
	}
	const T &operator[](unsigned int i) const{
		assert(i<this->n);
		return(el[i]);
	}
	Array<T> &operator+=(const Array<T> &rhs){
		if(rhs.n!=this->n){
			throw("trying to add arrays with different dimensions.\n");
		}
		for(int i=0;i<rhs.n;i++){
			this->el[i]+=rhs.el[i];
		}
		return *this;
	}
	friend Array<T> operator+(Array<T> lhs,const Array<T> &rhs){
		lhs+=rhs;
		return lhs;
	}
	Array<T> &operator=(const Array<T> &rhs){
/*
		printf("in operator=\n");
		this->el=rhs.el;
*/
		this->resize(rhs.n);
		for(int i=0;i<rhs.n;i++){
			this->el[i]=rhs.el[i];
		}
		this->n=rhs.n;
		return *this;
	}
	void print() const{
		int i=0;
		for(double x:el){
			printf("[%i]%.02f\t",i++,x);
		}
		printf("\n");
	}
	void rand(){
		int i;
		for(i=0;i<this->n;i++){
			this->el[i]=
			(random()-(RAND_MAX/2))*2.0/((double)RAND_MAX)/((double)RANDSCALING);
		}
	}
};

#endif
