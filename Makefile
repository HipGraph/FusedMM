BIN = ./bin
Kdir = ./kernels
KLIBdir = $(Kdir)/lib
KINCdir = $(Kdir)/include

# indextype : int64_t or int32_t
# NOTE: when comparing with MKL, use ibit=64 since we are using MKL_ILP64
ibit=64
#ibit=32

# valuetype precision : double single 
pre=
vlen=
#
# SIMD width on system: 
#    Update ARCH, SIMD variable in kernels/make.inc 
#    See kernels/include/simd.h for details    
#
# max dimension or max value of K for generated library 
mdim=128  
regblk=bacrb 
#regblk=acrb 
#regblk=crb 

kruntime=1
bestK=64    # needed when kruntime=1 

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
CC = g++
FLAGS = -fopenmp -O3 -march=native -std=c++11

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

all: $(BIN)/x$(pre)fusedMMtime_pt        
mkl: $(BIN)/x$(pre)fusedMMtime_MKL_pt        
#
#   serial version
#
$(BIN)/$(pre)fusedMMtime.o: fusedMMtime.cpp $(KINCdir)/kernels.h  
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(TYPFLAGS) -I$(KINCdir) -DCPP -c fusedMMtime.cpp -o $@   
$(BIN)/x$(pre)fusedMMtime: $(BIN)/$(pre)fusedMMtime.o $(sLIBS)  
	$(CC) $(FLAGS) -o $@ $^ $(sLIBS) -lm
#
#  parallel version 
#
$(BIN)/$(pre)fusedMMtime_pt.o: fusedMMtime.cpp $(KINCdir)/kernels.h  
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(TYPFLAGS) -I$(KINCdir) $(MYPT_FLAG) -DCPP \
		-c fusedMMtime.cpp -o $@   
$(BIN)/x$(pre)fusedMMtime_pt: $(BIN)/$(pre)fusedMMtime_pt.o $(ptLIBS)  
	$(CC) $(FLAGS) -o $@ $^ $(ptLIBS) -lm 

#
# ******** Build with mkl 
#
$(BIN)/$(pre)fusedMMtime_MKL_pt.o: fusedMMtime.cpp $(KINCdir)/kernels.h  
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(TYPFLAGS) -DTIME_MKL  -I$(KINCdir) $(PT_CC_MKL_FLAG) \
	   $(MYPT_FLAG) -DCPP -c fusedMMtime.cpp -o $@   

$(BIN)/x$(pre)fusedMMtime_MKL_pt: $(BIN)/$(pre)fusedMMtime_MKL_pt.o $(ptLIBS)  
	$(CC) $(FLAGS) -o $@ $^ $(ptLIBS) -lm $(PT_LD_MKL_FLAG) 

# ===========================================================================
# To generate FusedMM kernels 
# =========================================================================

$(sLIBS)  : $(ptLIBS)
$(ptLIBS) : $(Kdir)/rungen.sh  
	cd $(Kdir) ; ./rungen.sh -p $(pre) -i $(ibit) -s $(vlen) -e $(mdim) \
	   -v $(vlen) -t $(NTHREADS) -r $(regblk) -k $(kruntime) -b $(bestK)

#
# to generate datatset... not used yet 
#
gen-er:
	./scripts/gen_er.sh

gen-rmat:
	./scripts/gen_rmat.sh

download:
	./scripts/download.sh

#
# cleanup 
#

clean:
	rm -rf ./bin/*

killlib: 
	cd $(Kdir) ; make clean pre=s

