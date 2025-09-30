/*
 * Eric Ryan Montgomery
 * 08/26/2024
 * Program designed for...
 */
using namespace std;

//Build Cmd: g++ -fPIC -shared -o ctypeTest.so ctypesTest.cpp

//First Cmd: g++ -c -fPIC ctypesTest.cpp -o ctypesTest.o
//Second Cmd: g++ -shared -o ctypesTest.so ctypesTest.o

//Actual Class
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
	
};

//Wrappers
extern "C" {
	Coords *CreateCoords(int x, int y) {
		return new Coords(x, y);
	}
	
	void DeleteCoords(Coords *obj) {
		delete static_cast<Coords*>(obj);
	}
	
	void SetCoords(Coords *obj, int x, int y) {
		return static_cast<Coords*>(obj)->SetCoords(x, y);
	}
	
	int GetXCoord(Coords *obj) {
		return static_cast<Coords*>(obj)->GetXCoord();
	}
	
	int GetYCoord(Coords *obj) {
		return static_cast<Coords*>(obj)->GetYCoord();
	}
}

/*
extern "C" {
	int add(int x, int y) {
		return x + y;
	}
}
*/
