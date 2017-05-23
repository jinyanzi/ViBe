TARGETS= ViBe
CXX=g++
CXXFLAGS= `pkg-config opencv --cflags` -pg -Wall -std=c++11
LIBS=`pkg-config opencv --libs` -pg 

all: $(TARGETS)

ViBe:trajDebugger.h trajDebugger.cpp ViBe.h ViBe.cpp  ViBe_main.cpp 
	$(CXX) $(CXXFLAGS) trajDebugger.cpp ViBe.cpp ViBe_main.cpp -o ViBe $(LIBS) 
	
clean:
	rm ViBe *.o *.gch
