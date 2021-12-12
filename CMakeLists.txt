#This CMakeLists.txt file is for DGL integration
cmake_minimum_required(VERSION 3.0)

project(fusedmm C CXX)
#include_directories("include")
#change the flags based on architecture
set(CMAKE_C_FLAGS "-O2 -Wall -fPIC -O3 ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC -std=c++11 -O3 ${CMAKE_CXX_FLAGS}")
add_definitions(-DBETA0 -DVALUETYPE=float -DINDEXTYPE=int64_t -fopenmp -DPTTIME -DNTHREADS=48 -DLDB -DVLEN=16 -DBLC_ARCH -DBLC_X86)
FILE(GLOB ALLSOURCE *.c)
FILE(GLOB HEADERS *.h)
FILE(GLOB ALLOBJECT *.o)
add_library(fusedmm ${ALLOBJECT} ${HEADERS} ${ALLSOURCE})
set_target_properties(fusedmm PROPERTIES LINKER_LANGUAGE CXX)
