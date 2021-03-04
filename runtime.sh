#!/bin/bash

kstart=32
kend=1024
interval=64

#pre=s
#vlen=8
pre=s
vlen=16

Ddir=./dataset  # path 
#dataset="blogcatalog.mtx cora.mtx citeseer.mtx com-Amazon.mtx flickr.mtx pubmed.mtx youtube.mtx" 
dataset="harvard.mtx" 

rdir=./results

nr=5 # number of repeatation 
#nr=20 # number of repeatation 

kerns="t s m"  # t=tdist s=sigmoid m=spmm 
#rblk="bacrb acrb crb"  #bacrb = Z,Y,X blocking, 
rblk="bacrb"  #bacrb = Z,Y,X blocking, 

# last level of cache in KB, use lscpu 
cszKB=16000
# number of threads 
#nthd=10 
nthd=10

kruntime=0 # 1 = any value of K, provided bestK 
bestK=128

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
   make test pre=$pre vlen=$vlen NTHREADS=$nthd mdim=$kend ibit=64 regblk=$rb kruntime=$kruntime bestK=$bestK
   for kern in $kerns 
   do
      for dset in $dataset 
      do
         res=$rdir/${dset}-${kern}kern-${pre}real-${rb}.csv
         echo "Filename,NNZ,M,N,K,trusted-time,test-time,speedup" > $res
         for (( k=$kstart; k <= $kend; k=$k+$interval ))
         {
            ./bin/x${pre}fusedMMtime_pt -input $Ddir/$dset -K $k -nrep $nr -skHd 1 -t $kern -C ${cszKB} | tee -a ${res} 
         }
      done
   done
done
