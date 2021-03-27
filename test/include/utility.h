#ifndef _UTILITY_H
#define _UTILITY_H

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>

#ifdef ENABLE_LIBNUMA
   #include <numa.h>  /* to enable libnuma functionality */
#endif
// #include <tbb/scalable_allocator.h>

using namespace std;
#define EPSILON 0.001

template <class T>
struct ErrorTolerantEqual : public binary_function<T, T, bool> {
  ErrorTolerantEqual(const T &myepsilon) : epsilon(myepsilon){};
  inline bool operator()(const T &a, const T &b) const {
    // According to the IEEE 754 standard, negative zero and positive zero
    // should compare as equal with the usual (numerical) comparison operators,
    // like the == operators of C++

    if (a == b) // covers the "division by zero" case as well: max(a,b) can't be
                // zero if it fails
      return true; // covered the integral numbers case

    return (std::abs(a - b) < epsilon ||
            (std::abs(a - b) / max(std::abs(a), std::abs(b))) < epsilon);
  }
  T epsilon;
};

// Because identify reports ambiguity in PGI compilers
template <typename T> struct myidentity : public std::unary_function<T, T> {
  const T operator()(const T &x) const { return x; }
};

template <typename _ForwardIterator, typename _StrictWeakOrdering>
bool my_is_sorted(_ForwardIterator __first, _ForwardIterator __last,
                  _StrictWeakOrdering __comp) {
  if (__first == __last)
    return true;

  _ForwardIterator __next = __first;
  for (++__next; __next != __last; __first = __next, ++__next)
    if (__comp(*__next, *__first))
      return false;
  return true;
};

template <typename ITYPE> ITYPE CumulativeSum(ITYPE *arr, ITYPE size) {
  ITYPE prev;
  ITYPE tempnz = 0;
  for (ITYPE i = 0; i < size; ++i) {
    prev = arr[i];
    arr[i] = tempnz;
    tempnz += prev;
  }
  return (tempnz); // return sum
}

template <typename _ForwardIter, typename T>
void iota(_ForwardIter __first, _ForwardIter __last, T __value) {
  while (__first != __last)
    *__first++ = __value++;
}

template <typename T, typename I> T **allocate2D(I m, I n) {
  T **array = new T *[m];
  for (I i = 0; i < m; ++i)
    array[i] = new T[n];
  return array;
}

template <typename T, typename I> void deallocate2D(T **array, I m) {
  for (I i = 0; i < m; ++i)
    delete[] array[i];
  delete[] array;
}

template <typename T> struct absdiff : binary_function<T, T, T> {
  T operator()(T const &arg1, T const &arg2) const {
    using std::abs;
    return abs(arg1 - arg2);
  }
};

/* This function will return n % d.
   d must be one of: 1, 2, 4, 8, 16, 32, … */
inline unsigned int getModulo(unsigned int n, unsigned int d) {
  return (n & (d - 1));
}

// Same requirement (d=2^k) here as well
inline unsigned int getDivident(unsigned int n, unsigned int d) {
  while ((d = d >> 1))
    n = n >> 1;
  return n;
}

// Memory allocation by C++-new / Aligned malloc / scalable malloc
template <typename T> inline T *my_malloc(int array_size) {
// // #ifdef TBB
//   cout << "Called TBB" <<endl;
//   T *a = (T *)scalable_malloc(sizeof(T) * array_size);
//   for (int i=0; i<array_size; ++i)
//   {
//     a[i] = T();
//   }
//   return a;

// #ifdef CPP
// global_blockers[blocker_id] = static_cast<TripleNode*>(::operator new(SIZE * flops_by_row_blockers[blocker_id]));
  // T *a = static_cast<T*>(::operator new(array_size * sizeof(T)));
   T * a = new T[array_size];
  // T *a;
  // a = (T*) aligned_alloc(4096, array_size * sizeof(T));
// #pragma omp parallel
//   {
//       cout << "Take a look " << omp_get_num_threads() << endl;
//   }
  #pragma omp parallel for
  for(int i=0; i<array_size; i++)
  {
    a[i] = T();
  }
  return a;
// #elif defined IMM
//   return (T *)_mm_malloc(sizeof(T) * array_size, 64);
// #else
//   return (T *)scalable_malloc(sizeof(T) * array_size);
// #endif
}

// Memory deallocation
template <typename T> inline void my_free(T *a) {
#ifdef CPP
  delete[] a;
#elif defined IMM
  _mm_free(a);
#elif defined TBB
  scalable_free(a);
#else
  scalable_free(a);
#endif
}

// Prefix sum (Sequential)
template <typename T> void seq_scan(T *in, T *out, T N) {
  out[0] = 0;
  for (T i = 0; i < N - 1; ++i) {
    out[i + 1] = out[i] + in[i];
  }
}

// Prefix sum (Thread parallel)
template <typename T> void scan(T *in, T *out, T N) {
  // if the array is comparatively small, use sequential scan instead
  if (N < (1 << 17)) {
    seq_scan(in, out, N);
  } else {
    int tnum = 1;
#pragma omp parallel
    { tnum = omp_get_num_threads(); }
    T each_n = N / tnum;
    T *partial_sum = my_malloc<T>(tnum);
#pragma omp parallel
    {
      // thead level prefix summing
      int tid = omp_get_thread_num();
      T start = each_n * tid;
      T end = (tid < tnum - 1) ? start + each_n : N;
      out[start] = 0;
      for (T i = start; i < end - 1; ++i) {
        out[i + 1] = out[i] + in[i];
      }
      // calculate offset in every thread
      partial_sum[tid] = out[end - 1] + in[end - 1];
#pragma omp barrier

      T offset = 0;
      for (int i = 0; i < tid; ++i) {
        offset += partial_sum[i];
      }
      for (T i = start; i < end; ++i) {
        out[i] += offset;
      }
    }
    my_free<T>(partial_sum);
  }
}

// Sort by key
template <typename IT, typename NT>
inline void mergesort(IT *nnz_num, NT *nnz_sorting, IT *temp_num,
                      NT *temp_sorting, IT left, IT right) {
  int mid, i, j, k;

  if (left >= right) {
    return;
  }

  mid = (left + right) / 2;

  mergesort(nnz_num, nnz_sorting, temp_num, temp_sorting, left, mid);
  mergesort(nnz_num, nnz_sorting, temp_num, temp_sorting, mid + 1, right);

  for (i = left; i <= mid; ++i) {
    temp_num[i] = nnz_num[i];
    temp_sorting[i] = nnz_sorting[i];
  }

  for (i = mid + 1, j = right; i <= right; ++i, --j) {
    temp_sorting[i] = nnz_sorting[j];
    temp_num[i] = nnz_num[j];
  }

  i = left;
  j = right;

  for (k = left; k <= right; ++k) {
    if (temp_num[i] <= temp_num[j] && i <= mid) {
      nnz_num[k] = temp_num[i];
      nnz_sorting[k] = temp_sorting[i++];
    } else {
      nnz_num[k] = temp_num[j];
      nnz_sorting[k] = temp_sorting[j--];
    }
  }
}

// Sorting key-value
template <typename IT, typename NT>
inline void cpu_sorting_key_value(IT *key, NT *value, IT N) {
  IT *temp_key;
  NT *temp_value;

  temp_key = my_malloc<IT>(N);
  temp_value = my_malloc<NT>(N);

  mergesort(key, value, temp_key, temp_value, 0, N - 1);

  my_free<IT>(temp_key);
  my_free<NT>(temp_value);
}

// // query cpu cache
// size_t i386_cpuid_caches() {
//   int i;
//   size_t total_avail_cache = 0;

//   for (i = 0; i < 32; i++) {

//     // Variables to hold the contents of the 4 i386 legacy registers
//     uint32_t eax, ebx, ecx, edx;

//     eax = 4; // get cache info
//     ecx = i; // cache id

//     __asm__(
//         "cpuid"     // call i386 cpuid instruction
//         : "+a"(eax) // contains the cpuid command code, 4 for cache query
//           ,
//           "=b"(ebx), "+c"(ecx) // contains the cache id
//           ,
//           "=d"(edx)); // generates output in 4 registers eax, ebx, ecx and edx

//     // taken from http://download.intel.com/products/processor/manual/325462.pdf
//     // Vol. 2A 3-149
//     int cache_type = eax & 0x1F;

//     if (cache_type == 0) // end of valid cache identifiers
//       break;

//     char const *cache_type_string;
//     switch (cache_type) {
//     case 1:
//       cache_type_string = "Data Cache";
//       break;
//     case 2:
//       cache_type_string = "Instruction Cache";
//       break;
//     case 3:
//       cache_type_string = "Unified Cache";
//       break;
//     default:
//       cache_type_string = "Unknown Type Cache";
//       break;
//     }

//     int cache_level = (eax >>= 5) & 0x7;

//     int cache_is_self_initializing =
//         (eax >>= 3) & 0x1; // does not need SW initialization
//     int cache_is_fully_associative = (eax >>= 1) & 0x1;

//     // taken from http://download.intel.com/products/processor/manual/325462.pdf
//     // 3-166 Vol. 2A ebx contains 3 integers of 10, 10 and 12 bits respectively
//     unsigned int cache_sets = ecx + 1;
//     unsigned int cache_coherency_line_size = (ebx & 0xFFF) + 1;
//     unsigned int cache_physical_line_partitions = ((ebx >>= 12) & 0x3FF) + 1;
//     unsigned int cache_ways_of_associativity = ((ebx >>= 10) & 0x3FF) + 1;

//     // Total cache size is the product
//     size_t cache_total_size = cache_ways_of_associativity *
//                               cache_physical_line_partitions *
//                               cache_coherency_line_size * cache_sets;

//     if (cache_type == 1 or cache_type == 3)
//       total_avail_cache = std::max(total_avail_cache, cache_total_size);
//   }

//   return total_avail_cache;
// }

// inline void write(__int128 x)
// {
//     if(x<0)
//     {
//         putchar('-');
//         x=-x;
//     }
//     if(x>9)
//         write(x/10);
//     putchar(x%10+'0');
// }

#endif
