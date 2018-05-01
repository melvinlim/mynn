#include"idx.h"
void IDX::displayImage(double *img){
	double *p=img;
	for(int i=0;i<28;i++){
		for(int j=0;j<28;j++){
			if(*p++>0){
				printf(".");
			}else{
				printf(" ");
			}
		}
		printf("\n");
	}
}
void IDX::printImage(struct image *img){
	uint8_t *p=img->pixel;
	for(int i=0;i<28;i++){
		for(int j=0;j<28;j++){
			if(*p++>=128){
				printf(".");
			}else{
				printf(" ");
			}
		}
		printf("\n");
	}
}
Matrix *IDX::loadIDX(const char *filename){
	void *mem;
	struct idx2 *idx2Header;
	double *ptr;
	Matrix *mat;
	int rows,cols,matlen;
	int fd=open(filename,O_RDONLY);
	assert(fd>=0);
	mem=mmap(0,1024*1024,PROT_READ,MAP_FILE|MAP_SHARED,fd,0);
	idx2Header=(struct idx2 *)mem;
	assert(idx2Header!=MAP_FAILED);
	//assert(bswap_32(idx1Header->magic)==0x801);
	if((idx2Header->magic)!=0xe02){
		assert(bswap_32(idx2Header->magic)==0xe02);
		idx2Header->nRows=bswap_32(idx2Header->nRows);
		idx2Header->nCols=bswap_32(idx2Header->nCols);
		assert(false);	//untested.
	}
	rows=idx2Header->nRows;
	cols=idx2Header->nCols;
	matlen=rows*cols;
	mat=new Matrix(rows,cols);
	ptr=(double *)++idx2Header;
	for(int i=0;i<matlen;i++){
		mat->item[i]=*ptr++;
	}
	close(fd);
	assert(munmap(mem,1024*1024)==0);
	return mat;
}
void IDX::saveIDXEntry(Matrix *mat,int fd){
	struct idx2 hdr;
	hdr.magic=0xe02;
	hdr.nRows=mat->m;
	hdr.nCols=mat->n;
	double *ptr=mat->item;
	write(fd,&hdr,sizeof(hdr));
	for(int i=0;i<mat->m;i++){
		for(int j=0;j<mat->n;j++){
			write(fd,ptr++,sizeof(ptr));
		}
	}
}
void IDX::saveIDX(Matrix *mat,const char *filename){
	int fd=open(filename,O_CREAT|O_TRUNC|O_WRONLY);
	assert(fd>=0);
	saveIDXEntry(mat,fd);
	close(fd);
}
void IDX::saveNetwork(Net *network,const char *filename){
	int fd=open(filename,O_CREAT|O_TRUNC|O_WRONLY);
	assert(fd>=0);
	int layers=network->n;
	for(int i=0;i<layers;i++){
		saveIDXEntry(network->L[i]->mat,fd);
	}
	close(fd);
}
