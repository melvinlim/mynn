#ifndef _DEFS_H
#define _DEFS_H

//#define SOLVEXOR
//#define SOLVELINEAR
#define BATCH

//#define TESTGRAD
#define TESTSAVELOAD

#define LAMBDA_DECAY 0.0001

#define EPSILON 0.0001

#ifdef SOLVEXOR
	#define NINPUTS 2
	#define NOUTPUTS 2
	#define HIDDEN 15
	#define EPOCHS 100000
#elif defined SOLVELINEAR
	#define NINPUTS 3
	#define NOUTPUTS 1
	#define HIDDEN 15
	#define EPOCHS 100000
#else
	#define NINPUTS (28*28)
	#define NOUTPUTS (10)
	#define HIDDEN 200
	#define EPOCHS 2000
#endif
#define GAMMA (0.01)
#define RANDSCALING 10	//scale random weights to be from -0.1 to +0.1

#endif
