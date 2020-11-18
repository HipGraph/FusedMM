cmake_minimum_required(VERSION 3.0)

project(alldim)

include_directories("include")
#change the flags based on architecture
set(CMAKE_C_FLAGS "-O2 -Wall -O3 -mavx512f -mavx512dq ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC -std=c++11 -O3 -mavx512f -mavx512dq ${CMAKE_CXX_FLAGS}")
add_definitions(-Dibit=64 -DINDEXTYPE=int64_t -fopenmp -DPTTIME -DLDB -DVLEN=16 -DBLC_ARCH -DBLC_X86 -DBLC_AVX512)
FILE(GLOB ALLSOURCE src/*.c)
FILE(GLOB HEADERS ../simd/simd.h)
add_library(alldim ${HEADERS} ${ALLSOURCE})
