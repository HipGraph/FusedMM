# FusedMM

This is the official implementation of FusedMM method accepted for publication in IEEE IPDPS 2021 titled "FusedMM: A Unified SDDMM-SpMM Kernel for Graph Embedding and Graph Neural Networks". 

[**PDF is available in arXiv**](https://arxiv.org/abs/2011.06391)


## System Requirements
Users need to have the following software/tools installed in their PC/server. The source code was compiled and run successfully in Linux (Ubuntu and Debian distributions).
```
GCC version >= 4.9
OpenMP version >= 4.5
```
Some helpful links can be found at [GCC](https://gcc.gnu.org/install/), [OpenMP](https://clang-omp.github.io) and [Environment Setup](http://heather.cs.ucdavis.edu/~matloff/158/ToolsInstructions.html#compile_openmp).

## Compile and Run FusedMM

```
$ ./configure
```
It will probe the system and generate the Makefile and necessary configuration inside kernels for generated optimized kernels. This step will generate Makefile even if it fails to detect the hardware and users will be able to update the Makefile and kernels/Make.inc manually with appropriate hardware flags.  

To build the generated library for single precision float, use 
```
$ make
```
To test FusedMM for single precision float, use 
```
$ make test 
```

Compiling step will generate all executable files inside the bin folder. To run the tester and timer with a specific kernel using FusedMM, please use the following format:
```
./bin/xsOptFusedMMtime_fr_pt -input dataset/harvard.mtx 
```
The optimized kernels have the prefix `xsOptFusedMM*` and the generalized kernels have the prefix `xsFusedMM*`. There are several parameters which can be provided as follows:
```
-input <string>, full path of input file (required).
-K <int>, dimension of the embedding.
-C <int> Cachesize in KB to use cache flushing in timer
-nrep <int> Number of repetition in timer  
-T <1,0> want to run the tester along with timer  
```
## Download All Datasets of FusedMM ##
To conduct experiments using all the datasets of FusedMM paper, please download it from the following link: [**Datasets**](https://drive.google.com/drive/folders/1CktM59PBTVzSF8ekjU3EoYO5QDVrY7Yc?usp=sharing)

### Note for double precision floating point: 
Configure step detects the SIMD width for single precision. For double precision, it is normally half the width. Update "pre" (d for double) and "vlen" (SIMD width) in Makefile accordingly and use make command.

## Citation
If you find this repository helpful, please cite the following paper:
```
@inproceedings{rahman2020fusedmm,
  title={FusedMM: A Unified SDDMM-SpMM Kernel for Graph Embedding and Graph Neural Networks},
  author={Rahman, Md and Sujon, Majedul Haque and Azad, Ariful and others},
  booktitle={35th Proceedings of IEEE IPDPS},
  year={2021}
}
```

## Contact
Please contact the following person if you have any questions: Majedul Haque Sujon (`msujon@iu.edu`) or, Md. Khaledur Rahman (`morahma@iu.edu`).
