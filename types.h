#ifndef TYPES
#define TYPES

#define EPOCHS 1000
#define LAYERS 2
#define GAMMA (0.01)

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
	Array *delta;
};

struct Net{
	Layer **L;
	int size;
};

#endif
