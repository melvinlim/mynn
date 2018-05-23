#ifndef _DEFS_H
#define _DEFS_H

//#define SOLVEXOR
//#define SOLVELINEAR
#define BATCH

//#define TESTGRAD
#define TESTSAVELOAD

#ifdef SOLVEXOR
	#define NINPUTS 2
	#define NOUTPUTS 2
	#define HIDDEN 15
	#define EPOCHS 100000
	#define BATCHSIZE 4
#elif defined SOLVELINEAR
	#define NINPUTS 3
	#define NOUTPUTS 1
	#define HIDDEN 15
	#define EPOCHS 100000
	#define BATCHSIZE 8
#else
	#define NINPUTS (28*28)
	#define NOUTPUTS (10)
	#define HIDDEN 200
	#define EPOCHS 2000
	#define BATCHSIZE 10
#endif
#define RANDSCALING 10	//scale random weights to be from -0.1 to +0.1

#ifdef TESTGRAD
#undef BATCH
#define BATCH
#undef BATCHSIZE
#define BATCHSIZE 1
#endif

#define GAMMA (0.01/(float)BATCHSIZE)
#define LAMBDA_DECAY (0.0001/(float)BATCHSIZE)

#endif
