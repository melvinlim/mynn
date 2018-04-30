CXX =				g++
CXXFLAGS =	-std=c++11 -Wall -Wextra -g -c
LFLAGS =		-std=c++11 -Wall -Wextra -g -o mynn
OBJECTS =		main.o matrix.o array.o layer.o net.o data.o
mynn:		$(OBJECTS)
	$(CXX) $(LFLAGS) $(OBJECTS)
main:			main.cpp types.h defs.h
	$(CXX) $(CXXFLAGS) main.cpp
matrix:		matrix.cpp defs.h
	$(CXX) $(CXXFLAGS) matrix.cpp
array:		array.cpp defs.h
	$(CXX) $(CXXFLAGS) array.cpp
layer:		layer.cpp defs.h
	$(CXX) $(CXXFLAGS) layer.cpp
net:			net.cpp defs.h
	$(CXX) $(CXXFLAGS) net.cpp
data:			data.cpp defs.h
	$(CXX) $(CXXFLAGS) data.cpp
clean:
	rm -f mynn *.o a.out
