# FusedMM

A unified SDDMM-SPMM kernel 

run ./configure. It will probe the system and update the Makefile and necessary 
configuration inside kernels for generated optimized kernels. 

To build the generated library for single precision float, use 
make

To test FusedMM for single precision float, use 
make test 

Note for double precision floating point: 
Configure step detects the SIMD width for single precision. For double precision,
it normally half the width. Update "pre" and "vlen" in Makefile accordingly and
use make command. 
