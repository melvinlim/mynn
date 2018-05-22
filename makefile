CXX =				g++
CXXFLAGS =	-std=c++11 -Wall -Wextra -pedantic -g -c
OBJECTS =		layer.o net.o idx.o
all:			$(OBJECTS)
layer:		layer.cpp
	$(CXX) $(CXXFLAGS) layer.cpp
net:			net.cpp
	$(CXX) $(CXXFLAGS) net.cpp
idx:			idx.cpp
	$(CXX) $(CXXFLAGS) idx.cpp
clean:
	rm -f *.o
