/*
 *    CUDA implementation for fusedmm 
 */ 
#include<cassert>
#include<cstdio>
#define INDEXTYPE int 
#define VALUETYPE float


/*
 * NOTE: 
 *    Special kernel with following restriction:
 *      1. considering ldx = ldy = k = blockdim.x
 *      2. alpha = 1 and beta = 0 
 *      3. we are not using the val array of the adjMatrix 
 */

/*
 * NOTE: the idea is taken from NVIDIA's reduction slide 
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
   const INDEXTYPE m,      // rows of dense X and A matrix
   const INDEXTYPE n,      // rows of dense Y matrix and col of A matrix 
   const INDEXTYPE k,      // column dimension of X or X, d in paper  
   //const INDEXTYPE nnz,    // nonzeros: need to recreate csr with mkl 
   //const VALUETYPE *val,   // NNZ value  
   INDEXTYPE *rowptr,  // colids -> column indices 
   INDEXTYPE *colid, // starting index for rowptr
   VALUETYPE *x,     // Dense X matrix
   VALUETYPE *y,     // Dense Y matrix
   VALUETYPE *z           // Dense matrix z
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


   // compute for each nonzero in corresponding row, normally avg degree is small
   for (INDEXTYPE j = rowptr[i]; j < rowptr[i+1]; j++)
   {
      INDEXTYPE colidj = colid[j];
      INDEXTYPE jindex = colidj * blockDim.x;  // blockDim.x = k 
      VALUETYPE ry = y[jindex+id];
      VALUETYPE attrc = rx * ry;

#ifdef USE_WARP_SHUFFLE
      // TODO: possible to do it with warp shuffle and using less share mem 
      // need to compare timing results 
#else
      temp[tid] = attrc; // each thd in a block save its copy in share mem  
      __syncthreads();
      
      /*
       * NOTE: using similar idea as stated in NVIDIA's reduction slide
       */

      // reduceing to data in single warp 
      if (BLOCKSIZE >= 512)
      {
         if (tid < 256) temp[tid] += temp[tid+256]; 
         __syncthreads();
      }
      if (BLOCKSIZE >= 256)
      {
         if (tid < 128) temp[tid] += temp[tid+128]; 
         __syncthreads();
      }
      if (BLOCKSIZE >= 128)
      {
         if (tid < 64) temp[tid] += temp[tid+64]; 
         __syncthreads();
      }

      // reduction in first warp, no synch needed 
      if (tid < 32) warpReduce<BLOCKSIZE>(temp, tid);
      attrc = temp[0];  // already reduced in temp[0] 

#endif
   
      // After reduction, apply SOP ... for now, just scal it  by 0.5 
      //VALUETYPE d1 = fast_SM(attrc);
      VALUETYPE d1 = 0.5 * attrc;
      //rz += d1 * ry;
      rz += (1.0-d1) * ry;
   }
   z[id] = rz; // update z 
}


extern void sfusedMMcu_csr
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

/*
 *  first strategy: 
      1. consider 1024 >= k >= 128 and multiple of warp threads (32), time first 
      with k = 128, 256, 512 and 1024
      2. employ k threads in a block 
      3. grid dimension = total blocks = number of rows, m

      use const memory for indx, pntrn pntre, x and y???
 *    
 */
   
   // FIXME: use constant memory :  __const__ ... very limited storage!!!  
   // assumption: pntrb has m+1 element... so can be used as rowptr
   //VALUETYPE *d_m, *d_n, *d_k; // device copy of m, n, k variable  
   INDEXTYPE *d_rowptr, *d_colid;
   VALUETYPE *d_x, *d_y;
   VALUETYPE *d_z; 

   int nblocks = m;
   assert(k%32==0); // k is multiple of 32 threads (warp threads)
   int nthreads = k; 

/*
 * allocate in device memory
 *       NOTE, m,n,k is readonly values, don't want to copy it in memory... 
 *             rather use call by values 
 */
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
   switch(tkern)
   {
      case 't' :
         break;
      case 's' :
         // calling the GPU kernel
         switch(nthreads)
         {
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
               fprintf(stderr, "NOT supported for  K=nthreads=%d yet\n", k);
               break;
         }
         break;
      case 'm' :
         break;
      case 'g' :
         break;
   }

#ifdef DEBUG
   checkCudaErrors(cudaMemcpy(z, d_z, (m*ldz)*sizeof(VALUETYPE),
            cudaMemcpyDeviceToHost));
#else
   cudaMemcpy(z, d_z, (m*ldz)*sizeof(VALUETYPE),cudaMemcpyDeviceToHost);
#endif

cudaFree(d_rowptr);
cudaFree(d_colid);
cudaFree(d_x);
cudaFree(d_y);
cudaFree(d_z);
}
