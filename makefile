CXX =				g++
CXXFLAGS =	-std=c++11 -Wall -Wextra -pedantic -g -c
LFLAGS =		-std=c++11 -Wall -Wextra -pedantic -g -o nnet
OBJECTS =		main.o layer.o net.o data.o idx.o mnist.o xor.o linear.o
nnet:		$(OBJECTS)
	$(CXX) $(LFLAGS) $(OBJECTS)
main:			main.cpp types.h defs.h
	$(CXX) $(CXXFLAGS) main.cpp
layer:		layer.cpp
	$(CXX) $(CXXFLAGS) layer.cpp
net:			net.cpp
	$(CXX) $(CXXFLAGS) net.cpp
data:			data.cpp defs.h
	$(CXX) $(CXXFLAGS) data.cpp
idx:			idx.cpp
	$(CXX) $(CXXFLAGS) idx.cpp
mnist:		mnist.cpp
	$(CXX) $(CXXFLAGS) mnist.cpp
xor:			xor.cpp
	$(CXX) $(CXXFLAGS) xor.cpp
linear:		linear.cpp
	$(CXX) $(CXXFLAGS) linear.cpp
clean:
	rm -f nnet *.o a.out
