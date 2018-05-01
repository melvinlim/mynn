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
void IDX::loadIDX(Matrix *mat,const char *filename){
}
void IDX::saveIDX(Matrix *mat,const char *filename){
	char str[256];
	strncpy(str,filename,255);
	str[255]=0;
	int fd=open(str,O_CREAT|O_TRUNC|O_WRONLY);
	assert(fd>=0);
	struct idx3 hdr;
	hdr.magic=0x803;
	hdr.nImages=1;
	hdr.nRows=mat->m;
	hdr.nCols=mat->n;
	double *ptr=mat->item;
	write(fd,&hdr,sizeof(hdr));
	for(int i=0;i<mat->m;i++){
		for(int j=0;j<mat->n;j++){
			write(fd,ptr++,sizeof(ptr));
		}
	}
	close(fd);
}
/*
	void *mem1,*mem2;
	struct idx1 *idx1Header;
	struct idx3 *idx3Header;
	struct image *pImage;
	uint8_t *pLabel;
	int fd1=open("t10k-labels-idx1-ubyte",O_RDONLY);
	int fd3=open("t10k-images-idx3-ubyte",O_RDONLY);
	assert(fd1>=0);
	assert(fd3>=0);
	mem1=mmap(0,1024*1024,PROT_READ,MAP_FILE|MAP_SHARED,fd1,0);
	mem2=mmap(0,8*1024*1024,PROT_READ,MAP_FILE|MAP_SHARED,fd3,0);
	idx1Header=(struct idx1 *)mem1;
	idx3Header=(struct idx3 *)mem2;
	assert(idx1Header!=MAP_FAILED);
	assert(idx3Header!=MAP_FAILED);
	assert(bswap_32(idx1Header->magic)==0x801);
	assert(bswap_32(idx3Header->magic)==0x803);
	int nLabels=bswap_32(idx1Header->number);
	printf("0x%x\n%d\n",bswap_32(idx1Header->magic),bswap_32(idx1Header->number));
	printf("0x%x\n%d\n",bswap_32(idx3Header->magic),bswap_32(idx3Header->nImages));
	pLabel=(uint8_t *)(++idx1Header);
	pImage=(struct image *)(++idx3Header);
	sz=nLabels;
	pInputs=new Array *[sz];
	pOutputs=new Array *[sz];
	int nInputs=28*28;
	for(int i=0;i<nLabels;i++){
		pInputs[i]=new NetArray(pImage->pixel,nInputs);
		pOutputs[i]=new NetArray(*pLabel,10);
//		printf("label %d: %d\n",i,*pLabel);
//		IDX::printImage(pImage);
		pLabel++;
		pImage++;
	}
	close(fd1);
	close(fd3);
	assert(munmap(mem1,1024*1024)==0);
	assert(munmap(mem2,8*1024*1024)==0);
}
*/
