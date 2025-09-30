#include <pybind11/pybind11.h>
#include <iostream>
namespace py = pybind11;

//Build Cmd: g++ -fPIC -shared -o pybindTest.so pybindTest.cpp

//First Cmd: g++ -c -fPIC ctypesTest.cpp -o ctypesTest.o
//Second Cmd: g++ -shared -o ctypesTest.so ctypesTest.o

class Coords {
	private:
	int x;
	int y;
	
	public:
	//Constructor
	Coords(int x, int y) {
		this->x = x;
		this->y = y;
	}
	
	//Destructor
	~Coords() {}
	
	void SetCoords(int x, int y) {
		this->x = x;
		this->y = y;
	}
	
	int GetXCoord() {
		return x;
	}

	int GetYCoord() {
		return y;
	}
	
	void PrintCoords() {
		std::cout << x << ", " << y << "\n";
	}
	
};

int CHeavyComputation(int x) {
	int y = 0;
	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < x; ++j) {
			y = y + 1;
			if (y > 100) {
				y = 0;
			}
		}
	}
	return y;
}

PYBIND11_MODULE(pybindTestLibrary, handle) {
	py::class_<Coords>(handle, "PyCoords")
	.def(py::init<int, int>())
	.def("SetCoords", &Coords::SetCoords)
	.def("GetXCoord", &Coords::GetXCoord)
	.def("GetYCoord", &Coords::GetYCoord)
	.def("PrintCoords", &Coords::PrintCoords);
	
	handle.def("CHeavyComputation", &CHeavyComputation);
}
