my_mytarget: linear.cu
	nvcc -O1 linear.cu -o train -lm
