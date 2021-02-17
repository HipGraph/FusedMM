BIN = ./bin
Kdir = ./kernels
KLIBdir = $(Kdir)/lib
KINCdir = $(Kdir)/include

# indextype : int64_t or int32_t
# NOTE: when comparing with MKL, use ibit=64 since we are using MKL_ILP64
ibit=64
#ibit=32

# valuetype precision : double single 
pre=s

#
# SIMD width on system: 
#    Depend on ARCH, configure step sets SIMD variable in kernels/make.inc 
#    See kernels/include/simd.h for different width on different system  
#
vlen=8

#
#  Register blocking strategies: 
#  	bacrb = all three dense matrix register blocked
#  	acrb = X(A) and Z(C) register blocked
#  	crb = only Z(C) is register blocked 
#
regblk=bacrb 
#regblk=acrb 
#regblk=crb 

#
#   Two different phases: 
#   	K-compile time: kernel generated upto mdim but fully unrolled  
#   	K-runtime: kernel generated upto bestK unrolled and rolled beyond
#   Note: K-compile time mode only support kernels upto mdim, for arbitrary 
#   K, we should use K-runtime but after tuning the bestK 
#
#kruntime=0
mdim=1024 

kruntime=1   # 0 means K compile time, used in tuning phase  
bestK=512    # needed when kruntime=1, normally got from tuning step  

kern=s   # t = tdist/fr, s = sigmoid, m = spmm, g = gcn 
#setup flags based on type 
ifeq ($(pre), d)
   dtyp=-DDREAL
else
   dtyp=-DSREAL
endif
TYPFLAGS = -DINDEXTYPE=int$(ibit)_t -DINT$(ibit) $(dtyp)

# Library info  
sLIBS=$(KLIBdir)/$(pre)libgfusedMM_sequential.a 
ptLIBS=$(KLIBdir)/$(pre)libgfusedMM_pt.a 

KINCS=$(KINCdir)/kernels.h 

#
# tester/timer's compiler 
#
CC = gcc
CCFLAGS = -fopenmp -O3 -march=native 

CPP = g++
CPPFLAGS = -fopenmp -O3 -march=native -std=c++11

#
# My parallel flags 
#
ldb=l
NTHREADS=48
#NTHREADS=6
LDB=LOAD_BALANCE 
MYPT_FLAG = -DPTTIME -DNTHREADS=$(NTHREADS) -D$(LDB)  
#MYPT_FLAG = -DPTTIME -DNTHREADS=$(NTHREADS) -DSTATIC  

# ==========================================================================
#	Flags for MKL 
# =========================================================================

MKLROOT = /opt/intel/mkl
#
#parallel version of MKL 
#
PT_CC_MKL_FLAG = -DMKL_ILP64 -m64 -I${MKLROOT}/include
PT_LD_MKL_FLAG =  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
	       ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a \
	       ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp \
	       -lpthread -lm -ldl  

#serial version of MKL 
CC_MKL_FLAG =  -DMKL_ILP64 -m64 -I${MKLROOT}/include
LD_MKL_FLAG =  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
	       ${MKLROOT}/lib/intel64/libmkl_sequential.a \
	       ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread \
	       -lm -ldl 


# =========================================================================
# 	Target 
# =========================================================================

all: test 

data=dataset/blogcatalog.mtx
d=128  # feature dimension to test
t=s    # kernel to test, defaul sigmoid, other option = t, g, m
test: $(BIN)/x$(pre)fusedMMtime_pt       
	$(BIN)/x$(pre)fusedMMtime_pt -input $(data) -T 1 -K $(d) -t $(kern) 

time: $(BIN)/x$(pre)fusedMMtime_pt       
	$(BIN)/x$(pre)fusedMMtime_pt -input $(data) -K $(d)        

# MKL only supports SPMM kenrel 
test_mkl: $(BIN)/x$(pre)fusedMMtime_MKL_pt        
	$(BIN)/x$(pre)fusedMMtime_pt -input $(data) -T 1 -K $(d) -t m        
time_mkl: $(BIN)/x$(pre)fusedMMtime_MKL_pt        
	$(BIN)/x$(pre)fusedMMtime_pt -input $(data) -K $(d) -t m        

#
#   serial version
#
$(BIN)/$(pre)fusedMM.o: fusedMM.c fusedMM.h fusedMM_internal.h   
	mkdir -p $(BIN)
	$(CC) $(CCFLAGS) $(TYPFLAGS) -I$(KINCdir) -c fusedMM.c -o $@   
$(BIN)/$(pre)fusedMMtime.o: fusedMMtime.cpp fusedMM.h $(KINCdir)/kernels.h  
	mkdir -p $(BIN)
	$(CPP) $(CPPFLAGS) $(TYPFLAGS) -I$(KINCdir) -DCPP -c fusedMMtime.cpp -o $@   
$(BIN)/x$(pre)fusedMMtime: $(BIN)/$(pre)fusedMMtime.o  $(BIN)/$(pre)fusedMM.o $(sLIBS)  
	$(CPP) $(CPPFLAGS) -o $@ $^ $(sLIBS) -lm
#
#  parallel version 
#
$(BIN)/$(pre)fusedMM_pt.o: fusedMM.c fusedMM.h fusedMM_internal.h  
	mkdir -p $(BIN)
	$(CC) $(CCFLAGS) $(TYPFLAGS) -I$(KINCdir) $(MYPT_FLAG)  \
		-c fusedMM.c -o $@   
$(BIN)/$(pre)fusedMMtime_pt.o: test/fusedMMtime.cpp fusedMM.h $(KINCdir)/kernels.h  
	mkdir -p $(BIN)
	$(CPP) $(CPPFLAGS) $(TYPFLAGS) -I$(KINCdir) $(MYPT_FLAG) -DCPP \
		-c test/fusedMMtime.cpp -o $@   
$(BIN)/x$(pre)fusedMMtime_pt: $(BIN)/$(pre)fusedMMtime_pt.o $(BIN)/$(pre)fusedMM_pt.o $(ptLIBS)  
	$(CPP) $(CPPFLAGS) -o $@ $^ $(ptLIBS) -lm 
#
# ******** Build with mkl 
#
$(BIN)/$(pre)fusedMMtime_MKL_pt.o: test/fusedMMtime.cpp $(KINCdir)/kernels.h  
	mkdir -p $(BIN)
	$(CPP) $(CPPFLAGS) $(TYPFLAGS) -DTIME_MKL  -I$(KINCdir) $(PT_CC_MKL_FLAG) \
	   $(MYPT_FLAG) -DCPP -c test/fusedMMtime.cpp -o $@   

$(BIN)/x$(pre)fusedMMtime_MKL_pt: $(BIN)/$(pre)fusedMMtime_MKL_pt.o $(ptLIBS)  
	$(CPP) $(CPPFLAGS) -o $@ $^ $(ptLIBS) -lm $(PT_LD_MKL_FLAG) 

# ===========================================================================
# To generate FusedMM kernels 
# =========================================================================

$(sLIBS)  : $(ptLIBS)
$(ptLIBS) : $(Kdir)/rungen.sh  
	cd $(Kdir) ; ./rungen.sh -p $(pre) -i $(ibit) -s $(vlen) -e $(mdim) \
	   -v $(vlen) -t $(NTHREADS) -r $(regblk) -k $(kruntime) -b $(bestK)
#
# cleanup 
#

clean:
	rm -rf ./bin/*

killlib: 
	cd $(Kdir) ; make clean pre=s

