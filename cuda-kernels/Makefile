
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_75 --ptxas-options=-v --use_fast_math


cufusedmm.o:	fusedmm.cu Makefile
	nvcc fusedmm.cu -c -o cufusedmm.o $(INC) $(NVCCFLAGS)

fusedmm_cu.a:	fusedmm.cu Makefile
	nvcc fusedmm.cu -lib -o fusedmm_cu.a $(INC) $(NVCCFLAGS)
