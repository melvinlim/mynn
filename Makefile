all:
	nvcc mynn.cu -O0 -g -D _GLIBCXX_DEBUG -D _GLIBCXX_DEBUG_PEDANTIC -std=c++11 -lm
