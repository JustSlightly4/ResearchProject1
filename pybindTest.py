import sys
import pybindTestLibrary
import time

def PyHeavyComputation(x):
	y = 0
	for i in range(x):
		for j in range(x):
			y = y + 1
			if y > 100:
				y = 0
	return y

def main():
	#Python Version
	start = time.perf_counter()
	print(PyHeavyComputation(1))
	end = time.perf_counter()
	print(f"Python Execution time: {end - start:.6f} seconds")
	
	#Python Version
	start = time.perf_counter()
	print(PyHeavyComputation(99999))
	end = time.perf_counter()
	print(f"Python Execution time: {end - start:.6f} seconds\n\n")
	
	#C++ Version
	start = time.perf_counter()
	print(pybindTestLibrary.CHeavyComputation(1))
	end = time.perf_counter()
	print(f"C++ Execution time: {end - start:.6f} seconds")
	
	#C++ Version
	start = time.perf_counter()
	print(pybindTestLibrary.CHeavyComputation(99999))
	end = time.perf_counter()
	print(f"C++ Execution time: {end - start:.6f} seconds")
	
	return 0

main()
