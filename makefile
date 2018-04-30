CXX =				g++
CXXFLAGS =	-std=c++11 -Wall -g -c
LFLAGS =		-std=c++11 -Wall -g -o mynn
OBJECTS =		main.o
mynn:		$(OBJECTS)
	$(CXX) $(LFLAGS) $(OBJECTS)
main:			main.cpp types.h
	$(CXX) $(CXXFLAGS) main.cpp
clean:
	rm -f mynn *.o a.out
