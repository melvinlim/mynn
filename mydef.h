#ifndef MYDEF
#define MYDEF

#define MALLOC(x) (x *)malloc(sizeof(x))
#define MALLOC(x,n) (x *)malloc(n*sizeof(x))

#endif
