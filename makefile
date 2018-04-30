CXX =				g++
CXXFLAGS =	-std=c++11 -Wall -g -c
LFLAGS =		-std=c++11 -Wall -g -o mynn
OBJECTS =		main.o matrix.o array.o
mynn:		$(OBJECTS)
	$(CXX) $(LFLAGS) $(OBJECTS)
main:			main.cpp types.h defs.h
	$(CXX) $(CXXFLAGS) main.cpp
matrix:		matrix.cpp defs.h
	$(CXX) $(CXXFLAGS) matrix.cpp
array:		array.cpp defs.h
	$(CXX) $(CXXFLAGS) array.cpp
clean:
	rm -f mynn *.o a.out
