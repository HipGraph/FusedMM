/*
 *    1st version of CUDA implementation for fusedmm_sigmoid 
 *    
 *    Assumptions:
 *    =============
 *    1. Feature dimension (K) is multiple of 32 (warp-size). This kernel is NOT 
 *       optimized for smaller K, may perform well for K >= 512 since we 
 *       restricted the number of threads to be the feature dimension. That means,
 *       one block will work on single row. We will extend it to work on 
 *       multiple rows in next version which will improve the performance of
 *       smaller feature dimension K. Note that this kernel only works
 *       for K <= 1024.
 *    2. We assume that if the dense matrices are blocked, they are already copied
 *       in row-major format so that the lda is equal to the number of column.
 *    3. Implemented case: alpha = 1 and beta = 0 
 */ 
#include<cassert>
#include<cstdio>
#include <omp.h>
// to profile the code 
#ifdef ENABLE_PROFILING
   #include <cuda_profiler_api.h>
#endif
// to enable debug code 
//#define DEBUG 1 
#ifdef DEBUG 
   #include "helper_cuda.h"
#endif

#ifdef DREAL
   #define VALUETYPE double
#else
   #define VALUETYPE float
#endif

/*
 * NOTE: the idea behind of the reduction code is taken from NVIDIA's slide 
 *    Ref : https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 */

template <unsigned int BLOCKSIZE>
__device__ void warpReduce
(
   volatile VALUETYPE *sdata, // must use volatile to prevent compiler to use regs
   unsigned int tid
 )
{
   if (BLOCKSIZE >= 64)
      sdata[tid] += sdata[tid+32];
   if (BLOCKSIZE >= 32)
      sdata[tid] += sdata[tid+16];
   if (BLOCKSIZE >= 16)
      sdata[tid] += sdata[tid+8];
   if (BLOCKSIZE >= 8)
      sdata[tid] += sdata[tid+4];
   if (BLOCKSIZE >= 4)
      sdata[tid] += sdata[tid+2];
   if (BLOCKSIZE >= 2)
      sdata[tid] += sdata[tid+1];

}


template <unsigned int BLOCKSIZE>
__global__ void sfusedMMcu_sigmoid_a1b0_csr
(
   const INDEXTYPE m,         // rows of dense X and A matrix
   const INDEXTYPE n,         // rows of dense Y matrix and col of A matrix 
   const INDEXTYPE k,         // column dimension of X or X, d in paper  
   //const INDEXTYPE nnz,     // nonzeros: need to recreate csr with mkl 
   //const VALUETYPE *val,    // NNZ value  
   INDEXTYPE *rowptr,         // colids -> column indices 
   INDEXTYPE *colid,          // starting index for rowptr
   VALUETYPE *x,              // Dense X matrix
   VALUETYPE *y,              // Dense Y matrix
   VALUETYPE *z               // Dense matrix z
)
{
   INDEXTYPE i = blockIdx.x; // each block work on a single row 
   INDEXTYPE tid = threadIdx.x;
   INDEXTYPE id = threadIdx.x + blockDim.x*blockIdx.x; 
   VALUETYPE rx = x[id];


   //VALUETYPE rz = z[id];
   VALUETYPE rz = 0.0; // beta==0 case  

   // share memory to reduce the attrc force, size depends on nthreads 
   extern  __shared__  float temp[];
   temp[tid] = 0;

   // compute for each nonzero in corresponding row, normally avg degree is small
   for (INDEXTYPE j = rowptr[i]; j < rowptr[i+1]; j++)
   {
      INDEXTYPE colidj = colid[j];
      INDEXTYPE jindex = colidj * blockDim.x;  // blockDim.x = k 
      VALUETYPE ry = y[jindex+tid];
/*
 *    VOP operation: element wise multiplication 
 */
      VALUETYPE attrc = rx * ry;
/*
 *    ROP operation: sum reduction
 */
      temp[tid] = attrc; // each thd in a block save its copy in share mem  
      __syncthreads();
      
      // reduceing to data in single warp 
      if (BLOCKSIZE > 512)
      {
         if (tid < 512) temp[tid] += temp[tid+512]; 
         __syncthreads();
      }
      if (BLOCKSIZE > 256)
      {
         if (tid < 256) temp[tid] += temp[tid+256]; 
         __syncthreads();
      }
      if (BLOCKSIZE > 128)
      {
         if (tid < 128) temp[tid] += temp[tid+128]; 
         __syncthreads();
      }
      if (BLOCKSIZE > 64)
      {
         if (tid < 64) temp[tid] += temp[tid+64]; 
         __syncthreads();
      }

      // reduction in first warp, no synch needed 
      if (tid < 32) warpReduce<BLOCKSIZE>(temp, tid);
      
      __syncthreads();
      attrc = temp[0];  // already reduced in temp[0] 
/*
 *    SOP: need to apply sigmoid function later. Just to show proof of concept
 *        we are applying scaling 
 */
      //VALUETYPE d1 = exp(attrc);
      VALUETYPE d1 = 0.5 * attrc;
/*
 *    VSC + AOP operation 
 */
      //rz += d1 * ry;
      rz += (1.0-d1) * ry;
   }
/*
 * Update the output only once outside the loop 
 */
   z[id] = rz; // update z 
}

extern void fusedMMcu_csr
(
   const char tkern,       // 't' = tdist 's' = sigmoid 
   const INDEXTYPE m,      // rows of dense X matrix
   const INDEXTYPE n,      // rows of dense Y matrix 
   const INDEXTYPE k,      // column dimension of X or X, d in paper  
   const VALUETYPE alpha, 
   const INDEXTYPE nnz,    // nonzeros: need to recreate csr with mkl 
   const INDEXTYPE rows,   // number of rows... not needed 
   const INDEXTYPE cols,   // number of columns 
   const VALUETYPE *val,   // NNZ value  
   const INDEXTYPE *indx,  // colids -> column indices 
   const INDEXTYPE *pntrb, // starting index for rowptr
   const INDEXTYPE *pntre, // ending index for rowptr
   const VALUETYPE *x,     // Dense X matrix
   const INDEXTYPE ldx,    // 2nd dimension of x (col size since row-major)  
   const VALUETYPE *y,     // Dense Y matrix
   const INDEXTYPE ldy,    // 2nd dimension of y (col size since row-major)  
   const VALUETYPE beta,   // beta value 
   VALUETYPE *z,           // Dense matrix z
   const INDEXTYPE ldz     // 2nd dimension size of z (col size since row-major) 
)
{
cudaEvent_t startc, stopc;
cudaEventCreate(&startc);
cudaEventCreate(&stopc);
#ifdef ENABLE_PROFILING
   // starting the profiler 
   cudaProfilerStart(); 
#endif 

   INDEXTYPE *d_rowptr, *d_colid;
   VALUETYPE *d_x, *d_y;
   VALUETYPE *d_z; 

   int nblocks = m;
   assert(k%32==0); // k is multiple of 32 threads (warp threads)
   int nthreads = k; 
/*
 * beta is equal zero case for now 
 */
   assert(beta==0.0);

#ifdef DEBUG

   // allocate space for all arrays  
   checkCudaErrors(cudaMalloc((void **)&d_rowptr, (m+1)*sizeof(INDEXTYPE)));
   checkCudaErrors(cudaMemcpy(d_rowptr, pntrb, (m+1)*sizeof(INDEXTYPE),
            cudaMemcpyHostToDevice));
   
   checkCudaErrors(cudaMalloc((void **)&d_colid, (nnz)*sizeof(INDEXTYPE)));
   checkCudaErrors(cudaMemcpy(d_colid, indx, (nnz)*sizeof(INDEXTYPE),
            cudaMemcpyHostToDevice));

   assert(ldx==k);
   checkCudaErrors(cudaMalloc((void **)&d_x, (m*ldx)*sizeof(VALUETYPE)));   
   checkCudaErrors(cudaMemcpy(d_x, x, (m*ldx)*sizeof(VALUETYPE),
            cudaMemcpyHostToDevice));

   assert(ldy==k);
   checkCudaErrors(cudaMalloc((void **)&d_y, (n*ldy)*sizeof(VALUETYPE)));   
   checkCudaErrors(cudaMemcpy(d_y, y, (n*ldy)*sizeof(VALUETYPE),
            cudaMemcpyHostToDevice));

   // output  
   assert(ldz==k);
   checkCudaErrors(cudaMalloc((void **)&d_z, (m*ldz)*sizeof(VALUETYPE)));   
   
   checkCudaErrors(cudaMemcpy(d_z, z, (m*ldz)*sizeof(VALUETYPE),
            cudaMemcpyHostToDevice));

#else
   cudaMalloc((void **)&d_rowptr, (m+1)*sizeof(INDEXTYPE));
   cudaMemcpy(d_rowptr, pntrb, (m+1)*sizeof(INDEXTYPE),cudaMemcpyHostToDevice);

   cudaMalloc((void **)&d_colid, (nnz)*sizeof(INDEXTYPE));
   cudaMemcpy(d_colid, indx, (nnz)*sizeof(INDEXTYPE),cudaMemcpyHostToDevice);

   assert(ldx==k);
   cudaMalloc((void **)&d_x, (m*ldx)*sizeof(VALUETYPE));  // ldx == k 
   cudaMemcpy(d_x, x, (m*ldx)*sizeof(VALUETYPE),cudaMemcpyHostToDevice);

   assert(ldy==k);
   cudaMalloc((void **)&d_y, (n*ldy)*sizeof(VALUETYPE));  // ldy == k 
   cudaMemcpy(d_y, y, (n*ldy)*sizeof(VALUETYPE),cudaMemcpyHostToDevice);

   // output  
   assert(ldz==k);
   cudaMalloc((void **)&d_z, (m*ldz)*sizeof(VALUETYPE));   
#endif
   
   int shared_mem_size = sizeof(VALUETYPE) * nthreads;
   double start, end;
   // start = omp_get_wtime();
   cudaEventRecord(startc);
   switch(tkern)
   {
      case 't' :
         break;
      case 's' :
         // calling the GPU kernel
         //fprintf(stderr, "*********** calling sigmoid function\n");
         switch(nthreads)
         {
	    case 1024:
               sfusedMMcu_sigmoid_a1b0_csr<1024><<<nblocks,nthreads,shared_mem_size>>>
                  (m, n, k, d_rowptr, d_colid, d_x, d_y, d_z);
               break;
            case 512:
               sfusedMMcu_sigmoid_a1b0_csr<512><<<nblocks,nthreads,shared_mem_size>>>
                  (m, n, k, d_rowptr, d_colid, d_x, d_y, d_z); 
               break;
            case 256:
               sfusedMMcu_sigmoid_a1b0_csr<256><<<nblocks,nthreads,shared_mem_size>>>
                  (m, n, k, d_rowptr, d_colid, d_x, d_y, d_z); 
               break;
            case 128:
               sfusedMMcu_sigmoid_a1b0_csr<128><<<nblocks,nthreads,shared_mem_size>>>
                  (m, n, k, d_rowptr, d_colid, d_x, d_y, d_z); 
               break;
            case 64:
               sfusedMMcu_sigmoid_a1b0_csr<64><<<nblocks,nthreads,shared_mem_size>>>
                  (m, n, k, d_rowptr, d_colid, d_x, d_y, d_z); 
               break;
            case 32:
               sfusedMMcu_sigmoid_a1b0_csr<32><<<nblocks,nthreads,shared_mem_size>>>
                  (m, n, k, d_rowptr, d_colid, d_x, d_y, d_z); 
               break;
            case 16:
               sfusedMMcu_sigmoid_a1b0_csr<16><<<nblocks,nthreads,shared_mem_size>>>
                  (m, n, k, d_rowptr, d_colid, d_x, d_y, d_z); 
               break;
            case 8:
               sfusedMMcu_sigmoid_a1b0_csr<8><<<nblocks,nthreads,shared_mem_size>>>
                  (m, n, k, d_rowptr, d_colid, d_x, d_y, d_z); 
               break;
            case 4:
               sfusedMMcu_sigmoid_a1b0_csr<4><<<nblocks,nthreads,shared_mem_size>>>
                  (m, n, k, d_rowptr, d_colid, d_x, d_y, d_z); 
               break;
            case 2:
               sfusedMMcu_sigmoid_a1b0_csr<2><<<nblocks,nthreads,shared_mem_size>>>
                  (m, n, k, d_rowptr, d_colid, d_x, d_y, d_z); 
               break;
            case 1:
               sfusedMMcu_sigmoid_a1b0_csr<1><<<nblocks,nthreads,shared_mem_size>>>
                  (m, n, k, d_rowptr, d_colid, d_x, d_y, d_z); 
               break;
            default: 
               fprintf(stderr, "NOT supported for  K=nthreads=%ld yet\n", k);
               break;
         }
         break;
      case 'm' :
         break;
      case 'g' :
         break;
   }
   cudaEventRecord(stopc);
   cudaEventSynchronize(stopc);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, startc, stopc);
   // end = omp_get_wtime();
   // fprintf(stderr, "CUDA OMP TIME: %lf seconds\n", end - start); 
   fprintf(stderr, "CUDA Event TIME: %f seconds\n", milliseconds / 1000.0);
#ifdef DEBUG
   checkCudaErrors(cudaMemcpy(z, d_z, (m*ldz)*sizeof(VALUETYPE), cudaMemcpyDeviceToHost));
#else
   cudaMemcpy(z, d_z, (m*ldz)*sizeof(VALUETYPE),cudaMemcpyDeviceToHost);
#endif

cudaFree(d_z);
cudaFree(d_y);
cudaFree(d_x);
cudaFree(d_colid);
cudaFree(d_rowptr);

#ifdef ENABLE_PROFILING
   // stopping the profiler 
   cudaProfilerStop();
   cudaDeviceReset();
#endif

}
