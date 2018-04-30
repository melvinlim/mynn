#ifndef _DEFS_H
#define _DEFS_H

#define bswap_16(value) ((((value)&0xff)<<8)|((value)>>8))
#define bswap_32(value)	((bswap_16((value)&0xffff)<<16)|bswap_16((value)>>16))
//#define NINPUTS 2
//#define NOUTPUTS 2
#define NINPUTS 10
#define NOUTPUTS (28*28)
#define EPOCHS 100000
#define LAYERS 2
#define GAMMA (0.1)
#define RANDSCALING 10	//scale random weights to be from -0.1 to +0.1
#include<cstdint>

#pragma pack(push,1)
struct idx1{
	int32_t magic;
	int32_t number;
};
struct idx3{
	int32_t magic;
	int32_t nImages;
	int32_t nRows;
	int32_t nCols;
};
struct image{
	uint8_t pixel[28*28];
};
#pragma pack(pop)

#endif
