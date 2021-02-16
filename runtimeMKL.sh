#!/bin/bash

kstart=32
kend=256 
interval=16


trustedlib=_MKL

#pre=s
#vlen=8
pre=s
vlen=16

Ddir=../dataset  # path 
dataset="blogcatalog.mtx cora.mtx citeseer.mtx com-Amazon.mtx flickr.mtx pubmed.mtx youtube.mtx" 

rdir=./results

nr=20 # number of repeatation 

#kerns="t s m"  # t=tdist s=sigmoid m=spmm 
kerns="m"  # m=spmm only for MKL  
rblk="bacrb acrb crb"  #bacrb = Z,Y,X blocking, 

#nthd=10 
nthd=48 

kruntime=0 # 1 = any value of K, provided bestK 
bestK=64

usage="Usage: $0 [OPTION] ... 
Options: 
-v [val] 	value of vector width (vlen), see simd.h to find system vlen
-s [val]	starting value of dimension, must be multiple of vlen
-e [val]	ending value of dimension, must be multiple of vlen
-p [s,d]	precision of floating point, s for single precision, d for double precision
-i [32,64]	precision of int 
-k [s,t,m]	kernel  
-t [nthreads]	number of threads   
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
         kern=$OPTARG
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

for rb in $rblk 
do
   make clean
   make killlib
   make pre=$pre vlen=$vlen NTHREADS=$nthd mdim=$kend ibit=64 regblk=$rb kruntime=$kruntime bestK=$bestK
   # if want to run with MKL 
   make mkl pre=$pre ibit=64 NTHREADS=$nthd  
   for kern in $kerns 
   do
      for dset in $dataset 
      do
         res=$rdir/${dset}-${kern}kern-${pre}real-${rb}${trustedlib}.csv
         echo "Filename,NNZ,M,N,K,Trusted_inspect_time,Trusted_exe_time,Test_inspect_time,Test_exe_time,Speedup_exe_time,Speedup_total,Critical_point" > $res
         for (( k=$kstart; k <= $kend; k=$k+$interval ))
         {
            ./bin/x${pre}fusedMMtime${trustedlib}_pt -input $Ddir/$dset -K $k -nrep $nr -skHd 1 -t $kern | tee -a ${res}
         }
      done
   done
done
