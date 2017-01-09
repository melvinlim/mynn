#ifndef _TYPES
#define _TYPES

#define EPOCHS 1000
#define LAYERS 2
#define GAMMA (0.01)

// Thread block size
//#define BLOCK_SIZE 16
#define BLOCK_SIZE 2

#define RANDSCALING 10	//scale random weights to be from -0.1 to +0.1

template<typename T>
class Matrix{
public:
	int n;
	int m;
	std::vector<T> el;
	Matrix(){
		this->n=0;
		this->m=0;
	}
	Matrix(int n,int m){
		int i;
		this->n=n;
		this->m=m;
		el.resize(n*m);
		for(i=0;i<n*m;i++){
			el[i]=0;
		}
	}
	~Matrix(){
	}
	void resize(int n,int m){
		this->n=n;
		this->m=m;
		el.resize(n*m);
	}
	T &operator()(unsigned int i,unsigned int j){
		if(i>=this->n||j>=this->m){
			throw 0;
		}
		return(el[(i*(this->m))+j]);
	}
	const T &operator()(unsigned int i,unsigned int j) const{
		if(i>=this->n||j>=this->m){
			throw 0;
		}
		return(el[(i*(this->m))+j]);
	}
	void rand(){
		int i,j;
		for(i=0;i<this->n;i++){
			for(j=0;j<this->m;j++){
				(*this)(i,j)=
				(random()-(RAND_MAX/2))*2.0/((double)RAND_MAX)/((double)RANDSCALING);
			}
		}
	}
	void print(){
		int i,j;
		for(i=0;i<this->n;i++){
			for(j=0;j<this->m;j++){
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
		if(x){
			for(i=0;i<n;i++){
				this->el[i]=x[i];
			}
//			memcpy(p->el,x,n*sizeof(double));
		}
	}
	~Array(){
	}
	T &operator()(unsigned int i){
		if(i>=this->n){
			throw 0;
		}
		return(el[i]);
	}
	const T &operator()(unsigned int i) const{
		if(i>=this->n){
			throw 0;
		}
		return(el[i]);
	}
	T &operator[](unsigned int i){
		if(i>=this->n){
			throw 0;
		}
		return(el[i]);
	}
	const T &operator[](unsigned int i) const{
		if(i>=this->n){
			throw 0;
		}
		return(el[i]);
	}
	Array<T> &operator+=(const Array<T> &rhs){
		printf("in operator+=\n");
		for(int i=0;i<rhs.n;i++){
			this->el[i]+=rhs.el[i];
		}
		return *this;
	}
	friend Array<T> operator+(Array<T> lhs,const Array<T> &rhs){
		printf("in operator+\n");
		lhs+=rhs;
		return lhs;
	}
	Array<T> &operator=(const Array<T> &rhs){
		printf("in operator=\n");
/*
		for(int i=0;i<rhs.n;i++){
			this->el[i]=rhs.el[i];
		}
*/
		this->el=rhs.el;
		this->n=rhs.n;
		return *this;
	}
	void print(){
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
