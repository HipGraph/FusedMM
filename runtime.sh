#!/bin/bash

#
#  NOTE: Please run configure script and build the project by using make command
#  before running any timer scripts 
#

#
# NOTE: Kernel generation phases:  
# 1. When kruntime=0 (meaning, k compile time), the code generator will 
#  generate codes from kstart to kend by the interval. 
#     For example, kstart=16, kend=32, interval=4 will generate kernels for
#        K=16, K=20, K=24, K=28, K=32
#     NOTE: kstart, kend and interval must be mutilple of SIMD width or vlen
# 2. when kruntime=1 (meaning kruntime), the code will generate codes upto the 
#     bestK from kstart with the interval. However, K=bestK kernel will have 
#     a rolled loop to support all K >= bestK. We are testing different 
#     stratregies for kruntime using the codegen genkern_memorder.base  
#

kruntime=0   # 1 = K-runtime (supports all K values) , provided the bestK 
kstart=32
kend=256
interval=32
bestK=128    # best register blocking factor, needed when kruntime=1  

#
#  Register blocking strategies: 
#     bacrb = Use register blocking for all dense matrices: A(X), B(Y) and C(Z) 
#     acrb  = Use register blocking for A(X) and C(Z) dense matirces
#     crb   = Used register blocking only for C(Z) output matrix 
#

#rblk="bacrb acrb crb"  #bacrb = Z,Y,X blocking, 
rblk="bacrb"  #bacrb = Z,Y,X blocking, 

#
#  dataset with path
#

Ddir=./dataset  # path and dataset (without .mtx) 
#dataset="blogcatalog cora citeseer com-Amazon flickr pubmed youtube" 
dataset="harvard" 

#
#  setting path for result  
#

rdir=./results

#
#  Number of repetitions in timer  
#

nr=5  
#nr=20  

#
#  NOTE: We changed the way kerns works to incorporate user defined SOP for sigmoid
#           even in optFusedMM
#

kerns="g m s t"  # t=tdist s=sigmoid m=spmm g=gcn 
kernels="gcn spmm sigmoid tdist" 

#
#  cszKB = last level cache size in KB (use lscpu to find it out)
#  NOTE: When dataset is small enough to fit in the last level cache, we want to
#        use cache flushing timer. Otherwise, it will provide in-cache timing 
#        result for kernel and that is not appropriate from users/applications'
#        perspective. cszKB is needed for cache flushing timer. 
cszKB=16000

optF=Opt    # by default we are timing OptFusedMM 
pre=
px=s
nthd=
vlen=
IB=64

usage="Usage: $0 [OPTION] ... without any arguments, will run will default values 
Options: 
-v [val] 	SIMD width (vlen), Optional for single precision, already set in generated Makefile
-s [val]	starting value of dimension, must be multiple of vlen
-e [val]	ending value of dimension, must be multiple of vlen
-p [s,d]	precision of floating point, s for single precision, d for double precision
-i [32,64]	precision of int 
-k [s,t,m,g]	kernel  
-t [nthreads]	number of threads, optional if you want to use all the threads   
--help 		display help and exit 
"

while getopts "v:i:s:e:p:t:k:" opt
do
   case $opt in 
      v) 
         vlen=$OPTARG
         ;; 
      i) 
         IB=$OPTARG
         ;; 
      s) 
         kstart=$OPTARG
         ;; 
      e) 
         kend=$OPTARG
         ;; 
      p) 
         pre=$OPTARG
         ;;
      k) 
         kerns=$OPTARG
         ;;
      t) 
         nthd=$OPTARG
         ;;
      \?)
         echo "$usage"
         exit 1 
         ;;
   esac
done

mkdir -p $rdir 

if [ -n "$pre" ] ### not empty
then
   px=$pre   
   pre="pre=$pre"
fi

if [ -n "$vlen" ] ### not empty
then
   vlen="vlen=$vlen"
fi

if [ -n "$nthd" ] ### not empty
then
   nthd="NTHREADS=$nthd"
fi

#
# run the executables 
#

for rb in $rblk 
do
#
#  uncomment following lines if you want to rebuilt the project 
#
   #make clean
   #make killlib
   #echo "***** testing all implementation before timing"
   #make test $pre $vlen $nthd mdim=$kend ibit=$IB regblk=$rb kruntime=$kruntime bestK=$bestK
   
   for kern in $kerns 
   do
#
#     select executable, we created seperate exe to accomodate user_define func
#
      if [ "$kern" = s ] ## [[ "$kern" == s ]] 
      then
         kn=sigmoid
      elif [ "$kern" = t ] ## [[ "$kern" == t ]] 
      then 
         kn=tdist
      elif [ "$kern" = m ] ## [[ "$kern" == m ]] 
      then 
         kn=spmm
      elif [ "$kern" = g ] ## [[ "$kern" == g ]] 
      then 
         kn=gcn
      fi 
      
      exe=x${px}${optF}FusedMMtime_${kn}_pt

      for dset in $dataset 
      do
         res=$rdir/${dset}-${kn}-${px}real-${rb}.csv
         
         echo "Running $kn with the dataset $dset"
         echo "================================================="

         echo "Filename,NNZ,M,N,K,trusted-time,test-time,speedup" | tee $res
         for (( k=$kstart; k <= $kend; k=$k+$interval ))
         {
            ./bin/${exe} -input $Ddir/${dset}.mtx -K $k -nrep $nr -skHd 1 -C ${cszKB} | tee -a ${res} 
         }
      done
   done
done
