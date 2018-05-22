CXX =				g++
CXXFLAGS =	-std=c++11 -Wall -Wextra -pedantic -g -c
OBJECTS =		layer.o net.o idx.o
all:			$(OBJECTS)
layer:		layer.cpp defs.h
	$(CXX) $(CXXFLAGS) layer.cpp
net:			net.cpp defs.h
	$(CXX) $(CXXFLAGS) net.cpp
idx:			idx.cpp defs.h
	$(CXX) $(CXXFLAGS) idx.cpp
clean:
	rm -f *.o
