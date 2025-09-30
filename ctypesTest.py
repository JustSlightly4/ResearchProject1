import sys
import ctypes
import os


#Will keep WSL in mind
#Also, when I tried a smaller example lines 11 and 12 worked fine earlier

def main():
	
	#Imports the library
	path = os.getcwd()
	clibrary = ctypes.CDLL(os.path.join(path, 'ctypesTest.so'))

	#Constructor
	clibrary.CreateCoords.argtypes = [ctypes.c_int, ctypes.c_int]
	clibrary.CreateCoords.restype = c_void_p
	
	#Get X and Y coordinates
	clibrary.GetXCoord.argtypes = [c_void_p]
	clibrary.GetXCoord.restype = ctypes.c_int
	clibrary.GetYCoord.argtypes = [c_void_p]
	clibrary.GetYCoord.restype = ctypes.c_int

	#Uses function
	obj = clibrary.CreateCoords(1, 2)
	print(clibrary.GetXCoord(obj))
	
	return 0

main()
