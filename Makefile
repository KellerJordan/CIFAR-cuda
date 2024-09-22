my_mytarget: linear.cu
	nvcc -O2 linear.cu -o train -lm
