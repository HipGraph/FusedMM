# FusedMM

This is the official implementation of FusedMM method accepted for publication in IEEE IPDPS 2021 titled "FusedMM: A Unified SDDMM-SpMM Kernel for Graph Embedding and Graph Neural Networks". 

[**PDF is available in arXiv**](https://arxiv.org/abs/2011.06391)


## System Requirements
Users need to have the following softwares/tools installed in their PC/server. The source code was compiled and run successfully in both Linux and macOS.
```
GCC version >= 4.9
OpenMP version >= 4.5
```
Some helpful links can be found at [GCC](https://gcc.gnu.org/install/), [OpenMP](https://clang-omp.github.io) and [Environment Setup](http://heather.cs.ucdavis.edu/~matloff/158/ToolsInstructions.html#compile_openmp).

## Compile and Run FusedMM

```
$ ./configure
```
It will probe the system and update the Makefile and necessary configuration inside kernels for generated optimized kernels. 

To build the generated library for single precision float, use 
```
$ make
```
To test FusedMM for single precision float, use 
```
$ make test 
```

### Note for double precision floating point: 
Configure step detects the SIMD width for single precision. For double precision, it normally half the width. Update "pre" and "vlen" in Makefile accordingly and use make command.

## Citation
If you find this repository helpful, please cite the following paper:
```
@inproceedings{rahman2020fusedmm,
  title={FusedMM: A Unified SDDMM-SpMM Kernel for Graph Embedding and Graph Neural Networks},
  author={Rahman, Md and Sujon, Majedul Haque and Azad, Ariful and others},
  booktitle={Proceedings of IEEE IPDPS},
  year={2020}
}
```

## Contact
Please contact the following person if you have any questions: Majedul Haque Sujon (`msujon@iu.edu`) or, Md. Khaledur Rahman (`morahma@iu.edu`).
