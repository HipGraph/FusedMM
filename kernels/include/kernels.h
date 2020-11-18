#ifndef KERNEL_H
#define KERNEL_H
/*
 * Header file for API 
 */

#ifdef __cplusplus 
   extern "C"
   {
#endif
/* 
 * NOTE: You need to define INDEXTYPE as your appropriate int type, the kernel
 * implementation does not depend on int type 
 */

/* double precision function prototypes  */
void dgfusedMM_csr (const char tkern, const INDEXTYPE m, const INDEXTYPE n, 
      const INDEXTYPE k,const double alpha, const INDEXTYPE nnz, 
      const INDEXTYPE rows, const INDEXTYPE cols, const double *val, const INDEXTYPE *indx, 
      const INDEXTYPE *pntrb, const INDEXTYPE *pntre, const double *A, 
      const INDEXTYPE lda, const double *B, const INDEXTYPE ldb, 
      const double beta, double *C, const INDEXTYPE ldc);

void trusted_dgfusedMM_csr (const char tkern, const INDEXTYPE m, const INDEXTYPE n, 
      const INDEXTYPE k,const double alpha, const INDEXTYPE nnz, 
      const INDEXTYPE rows, const INDEXTYPE cols, const double *val, const INDEXTYPE *indx, 
      const INDEXTYPE *pntrb, const INDEXTYPE *pntre, const double *A, 
      const INDEXTYPE lda, const double *B, const INDEXTYPE ldb, 
      const double beta, double *C, const INDEXTYPE ldc);

/* single precision function prototypes  */
void sgfusedMM_csr (const char tkern, const INDEXTYPE m, const INDEXTYPE n, 
      const INDEXTYPE k,const float alpha, const INDEXTYPE nnz, 
      const INDEXTYPE rows, const INDEXTYPE cols, const float *val, const INDEXTYPE *indx, 
      const INDEXTYPE *pntrb, const INDEXTYPE *pntre, const float *A, 
      const INDEXTYPE lda, const float *B, const INDEXTYPE ldb, 
      const float beta, float *C, const INDEXTYPE ldc);

void trusted_sgfusedMM_csr (const char tkern, const INDEXTYPE m, const INDEXTYPE n, 
      const INDEXTYPE k,const float alpha, const INDEXTYPE nnz, 
      const INDEXTYPE rows, const INDEXTYPE cols, const float *val, const INDEXTYPE *indx, 
      const INDEXTYPE *pntrb, const INDEXTYPE *pntre, const float *A, 
      const INDEXTYPE lda, const float *B, const INDEXTYPE ldb, 
      const float beta, float *C, const INDEXTYPE ldc);

#ifdef __cplusplus 
   }  // extern "C"
#endif

#endif
