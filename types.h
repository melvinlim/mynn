#ifndef TYPES
#define TYPES

#define LAYERS 2
#define GAMMA (0.0001)

// Thread block size
//#define BLOCK_SIZE 16
#define BLOCK_SIZE 2

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride; 
    float* elements;
}Matrix;

typedef struct{
	int len;
	float *el;
}Array;

struct Layer{
	Array *in;
	Matrix *M;
	Matrix *dW;
	Array *out;
	Array *deriv;
};

struct Net{
	Layer **L;
	int size;
};

#endif
