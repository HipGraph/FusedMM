#ifdef __cplusplus
   extern "C"
   {
#endif
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#ifdef PTTIME
   #include<omp.h>
#endif
#include "../include/kernels.h"

#define Mjoin(pre,nam) my_join(pre, nam)
#define my_join(pre,nam) pre ## nam

#ifdef DREAL
   #include "../generated/include/dgmisc.h"
   #include "../generated/include/dgkernels_tdist.h"
   #include "../generated/include/dgkernels_sigmoid.h"
   #include "../generated/include/dgkernels_spmm.h"
   #include "../generated/include/dgkernels_gcn.h"
#else
   #include "../generated/include/sgmisc.h"
   #include "../generated/include/sgkernels_tdist.h"
   #include "../generated/include/sgkernels_sigmoid.h"
   #include "../generated/include/sgkernels_spmm.h"
   #include "../generated/include/sgkernels_gcn.h"
#endif

#ifdef DREAL 
   #define VALUETYPE double 
   #define PRE d
#else
   #define VALUETYPE float 
   #define PRE s
#endif
/* ============================================================================
 * Some trusted non-optimized kernel for the cases which are not handled by 
 * generated kernel. 
 * NOTE NOTE: in current version, optimized generated kernels can't be applied
 * when K%VLEN != 0. User should aways pad the row and make dimension multiple
 * of vector width before calling the optimized code.  
 *============================================================================*/

void trusted_fusedMM_tdist_csr 
(
   const char tkern,       /* 't' = tdist 's' = sigmoid 'm' = spmm 'g' = gcn */
   const INDEXTYPE m,      /* number of row of X */
   const INDEXTYPE n,      /* number of row of Y */
   const INDEXTYPE k,      /* dimension (col of X or Y) */ 
   const VALUETYPE alpha,  /* not used yet */ 
   const INDEXTYPE nnz,    /* nonzeros in sparse matrix  */
   const INDEXTYPE rows,   /* number of rows in sparse matrix */
   const INDEXTYPE cols,   /* number of columns in sparse matrix */
   const VALUETYPE *val,   /* value of NNZ  */
   const INDEXTYPE *indx,  /* colids -> column indices*/
   const INDEXTYPE *pntrb, /* starting index for rowptr */
   const INDEXTYPE *pntre, /* ending index for rowptr */
   const VALUETYPE *a,     /* Dense B matrix */
   const INDEXTYPE lda,    /* leading dimension of a (col size since row-major) */
   const VALUETYPE *b,     /* Dense B matrix */
   const INDEXTYPE ldb,    /* leading dimension of b (col size since row-major) */ 
   const VALUETYPE beta,   /* beta value */ 
   VALUETYPE *c,           /* Dense matrix c */
   const INDEXTYPE ldc     /* leading dimension size of c (col size since row-major) */ 
)
{
#if defined(PTTIME) && defined(LDB)
   omp_set_num_threads(NTHREADS);
   #pragma omp parallel
   {
      INDEXTYPE RowPerThd, tt;
      INDEXTYPE i, rowb, rowe;
      INDEXTYPE Mnnz = 0; /* non-zero count in M rows  */
      INDEXTYPE deg, cumRow, curRow;
      INDEXTYPE id = omp_get_thread_num();
      INDEXTYPE nthreads = omp_get_num_threads(); 
      
      for (i=0; i < m; i++)
         Mnnz += (pntre[i] - pntrb[i]); 
      RowPerThd = Mnnz / nthreads; 
      
      curRow = cumRow = 0; 
      tt = 1; 
      rowe = -1;  /* init */
      /* set rowstart for 1st thread */ 
      if (id == 0) 
         rowb = 0;
      for (i=0; i < m; i++)
      {
         deg = pntre[i] - pntrb[i]; 
         cumRow += deg;
         curRow += deg;
         if (curRow > RowPerThd)
         {
            if (tt == id)
               rowb = i; 
            else if (tt == id+1)
               rowe = i; 
            curRow = 0;
            RowPerThd = (Mnnz - cumRow) / (nthreads - tt);
            tt += 1; 
         }
      }
      if (tt == id+1)
         rowe = m; 

      for (i=rowb; i < rowe; i++)
#else /* not LBD or not PTTIME */
   #ifdef PTTIME
      #ifdef NTHREADS
      omp_set_num_threads(NTHREADS);
      #endif
      #ifdef DYNAMIC 
         #pragma omp parallel for schedule(dynamic)
      #else
         #pragma omp parallel for schedule(static)
      #endif
   #endif
   for (INDEXTYPE i = 0; i < m; i++)
#endif
   {
      const INDEXTYPE iindex = i * k;
      VALUETYPE T[k];
      for (INDEXTYPE j=pntrb[i]; j < pntre[i]; j++)
      {
         const INDEXTYPE cid = indx[j];
         const INDEXTYPE jindex = cid * k; 
         VALUETYPE attrc = 0.0;
         for (INDEXTYPE kk=0; kk < k; kk++)
         {
            T[k] = a[iindex + k] - b[jindex + k];
            attrc += T[k] * T[k];  
         }
         VALUETYPE d1 = -2.0 / (1.0 + attrc); 
         // update C 
         for (INDEXTYPE kk=0; kk < k; kk++)
         {
            VALUETYPE x = T[k] * d1 ;
            x = (x > SM_BOUND) ? SM_BOUND : x; 
            x = (x < -SM_BOUND) ? -SM_BOUND : x; 
            c[iindex+k] = c[iindex+k]  + x;
         }
      }
   }
#if defined(PTTIME) && defined(LDB)
   }
#endif 
}

void trusted_fusedMM_sigmoid_csr 
(
   const char tkern,       /* 't' = tdist 's' = sigmoid 'm' = spmm 'g' = gcn */
   const INDEXTYPE m,      /* number of row of X */
   const INDEXTYPE n,      /* number of row of Y */
   const INDEXTYPE k,      /* dimension (col of X or Y) */ 
   const VALUETYPE alpha,  /* not used yet */ 
   const INDEXTYPE nnz,    /* nonzeros in sparse matrix  */
   const INDEXTYPE rows,   /* number of rows in sparse matrix */
   const INDEXTYPE cols,   /* number of columns in sparse matrix */
   const VALUETYPE *val,   /* value of NNZ  */
   const INDEXTYPE *indx,  /* colids -> column indices*/
   const INDEXTYPE *pntrb, /* starting index for rowptr */
   const INDEXTYPE *pntre, /* ending index for rowptr */
   const VALUETYPE *a,     /* Dense B matrix */
   const INDEXTYPE lda,    /* leading dimension of a (col size since row-major) */
   const VALUETYPE *b,     /* Dense B matrix */
   const INDEXTYPE ldb,    /* leading dimension of b (col size since row-major) */ 
   const VALUETYPE beta,   /* beta value */ 
   VALUETYPE *c,           /* Dense matrix c */
   const INDEXTYPE ldc     /* leading dimension size of c (col size since row-major) */ 
)
{
#ifndef SOP_INHOUSE
   extern int SOP_UDEF_FUNC(VALUETYPE val, VALUETYPE *out);
#else
   VALUETYPE *sm_table = (VALUETYPE*)malloc(sizeof(VALUETYPE)*SM_TABLE_SIZE);
   if (!sm_table)
   {
      fprintf(stderr, 
            "Unable to allocate memory for SM TABLE in trusted kernel!!!\n");
      exit(1);
   }
#endif
#if defined(PTTIME) && defined(LDB)
   omp_set_num_threads(NTHREADS);
   #pragma omp parallel
   {
      INDEXTYPE RowPerThd, tt;
      INDEXTYPE i, rowb, rowe;
      INDEXTYPE Mnnz = 0; /* non-zero count in M rows  */
      INDEXTYPE deg, cumRow, curRow;
      INDEXTYPE id = omp_get_thread_num();
      INDEXTYPE nthreads = omp_get_num_threads(); 
      
      for (i=0; i < m; i++)
         Mnnz += (pntre[i] - pntrb[i]); 
      RowPerThd = Mnnz / nthreads; 
      
      curRow = cumRow = 0; 
      tt = 1; 
      rowe = -1;  /* init */
      /* set rowstart for 1st thread */ 
      if (id == 0) 
         rowb = 0;
      for (i=0; i < m; i++)
      {
         deg = pntre[i] - pntrb[i]; 
         cumRow += deg;
         curRow += deg;
         if (curRow > RowPerThd)
         {
            if (tt == id)
               rowb = i; 
            else if (tt == id+1)
               rowe = i; 
            curRow = 0;
            RowPerThd = (Mnnz - cumRow) / (nthreads - tt);
            tt += 1; 
         }
      }
      if (tt == id+1)
         rowe = m; 

      for (i=rowb; i < rowe; i++)
#else /* not LBD or not PTTIME */
   #ifdef PTTIME
      #ifdef NTHREADS
      omp_set_num_threads(NTHREADS);
      #endif
      #ifdef DYNAMIC 
         #pragma omp parallel for schedule(dynamic)
      #else
         #pragma omp parallel for schedule(static)
      #endif
   #endif
   for (INDEXTYPE i = 0; i < m; i++)
#endif
   {
      const INDEXTYPE iindex = i * k;
      for (INDEXTYPE j=pntrb[i]; j < pntre[i]; j++)
      {
         const INDEXTYPE cid = indx[j];
         const INDEXTYPE jindex = cid * k; 
         VALUETYPE attrc = 0.0;
         VALUETYPE d1; 
         for (INDEXTYPE kk=0; kk < k; kk++)
            attrc += a[iindex+kk] * b[jindex+kk];
   #ifdef SOP_INHOUSE
         d1 = fast_SM(attrc, sm_table);
         d1 = 1.0 - d1;
   #else
         SOP_UDEF_FUNC(attrc, &d1);
   #endif
         // update C 
         for (INDEXTYPE kk=0; kk < k; kk++)
            c[iindex+kk] += d1*b[jindex+kk];
      }
   }
#if defined(PTTIME) && defined(LDB)
   }
#endif
#ifdef SOP_INHOUSE
   free(sm_table);
#endif
}

void trusted_fusedMM_spmm_csr 
(
   const char tkern,       /* 't' = tdist 's' = sigmoid 'm' = spmm 'g' = gcn */
   const INDEXTYPE m,      /* number of row of X */
   const INDEXTYPE n,      /* number of row of Y */
   const INDEXTYPE k,      /* dimension (col of X or Y) */ 
   const VALUETYPE alpha,  /* not used yet */ 
   const INDEXTYPE nnz,    /* nonzeros in sparse matrix  */
   const INDEXTYPE rows,   /* number of rows in sparse matrix */
   const INDEXTYPE cols,   /* number of columns in sparse matrix */
   const VALUETYPE *val,   /* value of NNZ  */
   const INDEXTYPE *indx,  /* colids -> column indices*/
   const INDEXTYPE *pntrb, /* starting index for rowptr */
   const INDEXTYPE *pntre, /* ending index for rowptr */
   const VALUETYPE *a,     /* Dense B matrix */
   const INDEXTYPE lda,    /* leading dimension of a (col size since row-major) */
   const VALUETYPE *b,     /* Dense B matrix */
   const INDEXTYPE ldb,    /* leading dimension of b (col size since row-major) */ 
   const VALUETYPE beta,   /* beta value */ 
   VALUETYPE *c,           /* Dense matrix c */
   const INDEXTYPE ldc     /* leading dimension size of c (col size since row-major) */ 
)
{
#if defined(PTTIME) && defined(LDB)
   omp_set_num_threads(NTHREADS);
   #pragma omp parallel
   {
      INDEXTYPE RowPerThd, tt;
      INDEXTYPE i, rowb, rowe;
      INDEXTYPE Mnnz = 0; /* non-zero count in M rows  */
      INDEXTYPE deg, cumRow, curRow;
      INDEXTYPE id = omp_get_thread_num();
      INDEXTYPE nthreads = omp_get_num_threads(); 
      
      for (i=0; i < m; i++)
         Mnnz += (pntre[i] - pntrb[i]); 
      RowPerThd = Mnnz / nthreads; 
      
      curRow = cumRow = 0; 
      tt = 1; 
      rowe = -1;  /* init */
      /* set rowstart for 1st thread */ 
      if (id == 0) 
         rowb = 0;
      for (i=0; i < m; i++)
      {
         deg = pntre[i] - pntrb[i]; 
         cumRow += deg;
         curRow += deg;
         if (curRow > RowPerThd)
         {
            if (tt == id)
               rowb = i; 
            else if (tt == id+1)
               rowe = i; 
            curRow = 0;
            RowPerThd = (Mnnz - cumRow) / (nthreads - tt);
            tt += 1; 
         }
      }
      if (tt == id+1)
         rowe = m; 

      for (i=rowb; i < rowe; i++)
#else /* not LBD or not PTTIME */
   #ifdef PTTIME
      #ifdef NTHREADS
      omp_set_num_threads(NTHREADS);
      #endif
      #ifdef DYNAMIC 
         #pragma omp parallel for schedule(dynamic)
      #else
         #pragma omp parallel for schedule(static)
      #endif
   #endif
   for (INDEXTYPE i = 0; i < m; i++)
#endif
   {
      const INDEXTYPE iindex = i * k;
      for (INDEXTYPE j=pntrb[i]; j < pntre[i]; j++)
      {
         const INDEXTYPE cid = indx[j];
         const INDEXTYPE jindex = cid * k; 
         VALUETYPE v0 = val[j];
         // update C 
         for (INDEXTYPE kk=0; kk < k; kk++)
            c[iindex+kk] += v0 * b[jindex+kk];
      }
   }
#if defined(PTTIME) && defined(LDB)
   }
#endif 
}

void trusted_fusedMM_gcn_csr 
(
   const char tkern,       /* 't' = tdist 's' = sigmoid 'm' = spmm 'g' = gcn */
   const INDEXTYPE m,      /* number of row of X */
   const INDEXTYPE n,      /* number of row of Y */
   const INDEXTYPE k,      /* dimension (col of X or Y) */ 
   const VALUETYPE alpha,  /* not used yet */ 
   const INDEXTYPE nnz,    /* nonzeros in sparse matrix  */
   const INDEXTYPE rows,   /* number of rows in sparse matrix */
   const INDEXTYPE cols,   /* number of columns in sparse matrix */
   const VALUETYPE *val,   /* value of NNZ  */
   const INDEXTYPE *indx,  /* colids -> column indices*/
   const INDEXTYPE *pntrb, /* starting index for rowptr */
   const INDEXTYPE *pntre, /* ending index for rowptr */
   const VALUETYPE *a,     /* Dense B matrix */
   const INDEXTYPE lda,    /* leading dimension of a (col size since row-major) */
   const VALUETYPE *b,     /* Dense B matrix */
   const INDEXTYPE ldb,    /* leading dimension of b (col size since row-major) */ 
   const VALUETYPE beta,   /* beta value */ 
   VALUETYPE *c,           /* Dense matrix c */
   const INDEXTYPE ldc     /* leading dimension size of c (col size since row-major) */ 
)
{
#if defined(PTTIME) && defined(LDB)
   omp_set_num_threads(NTHREADS);
   #pragma omp parallel
   {
      INDEXTYPE RowPerThd, tt;
      INDEXTYPE i, rowb, rowe;
      INDEXTYPE Mnnz = 0; /* non-zero count in M rows  */
      INDEXTYPE deg, cumRow, curRow;
      INDEXTYPE id = omp_get_thread_num();
      INDEXTYPE nthreads = omp_get_num_threads(); 
      
      for (i=0; i < m; i++)
         Mnnz += (pntre[i] - pntrb[i]); 
      RowPerThd = Mnnz / nthreads; 
      
      curRow = cumRow = 0; 
      tt = 1; 
      rowe = -1;  /* init */
      /* set rowstart for 1st thread */ 
      if (id == 0) 
         rowb = 0;
      for (i=0; i < m; i++)
      {
         deg = pntre[i] - pntrb[i]; 
         cumRow += deg;
         curRow += deg;
         if (curRow > RowPerThd)
         {
            if (tt == id)
               rowb = i; 
            else if (tt == id+1)
               rowe = i; 
            curRow = 0;
            RowPerThd = (Mnnz - cumRow) / (nthreads - tt);
            tt += 1; 
         }
      }
      if (tt == id+1)
         rowe = m; 

      for (i=rowb; i < rowe; i++)
#else /* not LBD or not PTTIME */
   #ifdef PTTIME
      #ifdef NTHREADS
      omp_set_num_threads(NTHREADS);
      #endif
      #ifdef DYNAMIC 
         #pragma omp parallel for schedule(dynamic)
      #else
         #pragma omp parallel for schedule(static)
      #endif
   #endif
   for (INDEXTYPE i = 0; i < m; i++)
#endif
   {
      const INDEXTYPE iindex = i * k;
      for (INDEXTYPE j=pntrb[i]; j < pntre[i]; j++)
      {
         const INDEXTYPE cid = indx[j];
         const INDEXTYPE jindex = cid * k; 
         // update C 
         for (INDEXTYPE kk=0; kk < k; kk++)
            c[iindex+kk] += b[jindex+kk];
      }
   }
#if defined(PTTIME) && defined(LDB)
   }
#endif 
}

//void Mjoin(PRE,fusedMM_csr) 
#ifdef DREAL 
void dgfusedMM_csr
#else
void sgfusedMM_csr
#endif
(
   const char tkern,       /* 't' = tdist 's' = sigmoid 'm' = spmm 'g' = gcn */
   const INDEXTYPE m,      /* number of row of X */
   const INDEXTYPE n,      /* number of row of Y */
   const INDEXTYPE k,      /* dimension (col of X or Y) */ 
   const VALUETYPE alpha,  /* not used yet */ 
   const INDEXTYPE nnz,    /* nonzeros in sparse matrix  */
   const INDEXTYPE rows,   /* number of rows in sparse matrix */
   const INDEXTYPE cols,   /* number of columns in sparse matrix */
   const VALUETYPE *val,   /* value of NNZ  */
   const INDEXTYPE *indx,  /* colids -> column indices*/
   const INDEXTYPE *pntrb, /* starting index for rowptr */
   const INDEXTYPE *pntre, /* ending index for rowptr */
   const VALUETYPE *a,     /* Dense B matrix */
   const INDEXTYPE lda,    /* leading dimension of a (col size since row-major) */
   const VALUETYPE *b,     /* Dense B matrix */
   const INDEXTYPE ldb,    /* leading dimension of b (col size since row-major) */ 
   const VALUETYPE beta,   /* beta value */ 
   VALUETYPE *c,           /* Dense matrix c */
   const INDEXTYPE ldc     /* leading dimension size of c (col size since row-major) */ 
)
{
   INDEXTYPE kk;
   
   switch(tkern)
   {
      case 't': // tdist
         if (KRUNTIME_TDIST && k >= BESTK_TDIST)
            kk = BESTK_TDIST/GVLEN; /* GVLEN: generated kernels vlen */
         else
         {
            kk = k / GVLEN;
            if (k % GVLEN || k > MAXDIM_TDIST) /* no optimize kernel */
            {
               trusted_fusedMM_tdist_csr(tkern, m, n, k, alpha, nnz, rows,
                                        cols, val, indx, pntrb, pntre, a, lda, 
                                        b, ldb, beta, c,ldc);
               return;
            }
         }
         if (beta == 0)
         #ifdef DREAL 
            dgenkernels_tdist_b0[kk-1](tkern, m, n, k, alpha, nnz, rows,
                                       cols, val, indx, pntrb, pntre, a, lda, 
                                       b, ldb, beta, c, ldc);
         #else
            sgenkernels_tdist_b0[kk-1](tkern, m, n, k, alpha, nnz, rows,
                                       cols, val, indx, pntrb, pntre, a, lda, 
                                       b, ldb, beta, c, ldc);
         #endif
         else /* beta == 1 */
         #ifdef DREAL 
            dgenkernels_tdist_b1[kk-1](tkern, m, n, k, alpha, nnz, rows,
                                       cols, val, indx, pntrb, pntre, a, lda, 
                                       b, ldb, beta, c, ldc);
         #else
            sgenkernels_tdist_b1[kk-1](tkern, m, n, k, alpha, nnz, rows,
                                       cols, val, indx, pntrb, pntre, a, lda, 
                                       b, ldb, beta, c, ldc);
         #endif
         break;
      case 's': // sigmoid
         if (KRUNTIME_SIGMOID && k >= BESTK_SIGMOID)
            kk = BESTK_SIGMOID/GVLEN; /* GVLEN: generated kernels vlen */
         else
         {
            kk = k / GVLEN;
            if (k % GVLEN || k > MAXDIM_SIGMOID) /* no optimize kernel */
            {
               trusted_fusedMM_sigmoid_csr(tkern, m, n, k, alpha, nnz, rows,
                                        cols, val, indx, pntrb, pntre, a, lda, 
                                        b, ldb, beta, c,ldc);
               return;
            }
         }
         if (beta == 0)
         #ifdef DREAL 
            dgenkernels_sigmoid_b0[kk-1](tkern, m, n, k, alpha, nnz, rows,
                                       cols, val, indx, pntrb, pntre, a, lda, 
                                       b, ldb, beta, c, ldc);
         #else
            sgenkernels_sigmoid_b0[kk-1](tkern, m, n, k, alpha, nnz, rows,
                                       cols, val, indx, pntrb, pntre, a, lda, 
                                       b, ldb, beta, c, ldc);
         #endif
         else /* beta == 1 */
         #ifdef DREAL 
            dgenkernels_sigmoid_b1[kk-1](tkern, m, n, k, alpha, nnz, rows,
                                       cols, val, indx, pntrb, pntre, a, lda, 
                                       b, ldb, beta, c, ldc);
         #else
            sgenkernels_sigmoid_b1[kk-1](tkern, m, n, k, alpha, nnz, rows,
                                       cols, val, indx, pntrb, pntre, a, lda, 
                                       b, ldb, beta, c, ldc);
         #endif
         break;
      case 'm': // spmm
         if (KRUNTIME_SPMM && k >= BESTK_SPMM)
            kk = BESTK_SPMM/GVLEN; /* GVLEN: generated kernels vlen */
         else
         {
            kk = k / GVLEN;
            if (k % GVLEN || k > MAXDIM_SPMM) /* no optimize kernel */
            {
               trusted_fusedMM_spmm_csr(tkern, m, n, k, alpha, nnz, rows,
                                        cols, val, indx, pntrb, pntre, a, lda, 
                                        b, ldb, beta, c,ldc);
               return;
            }
         }
         if (beta == 0)
         #ifdef DREAL 
            dgenkernels_spmm_b0[kk-1](tkern, m, n, k, alpha, nnz, rows,
                                       cols, val, indx, pntrb, pntre, a, lda, 
                                       b, ldb, beta, c, ldc);
         #else
            sgenkernels_spmm_b0[kk-1](tkern, m, n, k, alpha, nnz, rows,
                                       cols, val, indx, pntrb, pntre, a, lda, 
                                       b, ldb, beta, c, ldc);
         #endif
         else /* beta == 1 */
         #ifdef DREAL 
            dgenkernels_spmm_b1[kk-1](tkern, m, n, k, alpha, nnz, rows,
                                       cols, val, indx, pntrb, pntre, a, lda, 
                                       b, ldb, beta, c, ldc);
         #else
            sgenkernels_spmm_b1[kk-1](tkern, m, n, k, alpha, nnz, rows,
                                       cols, val, indx, pntrb, pntre, a, lda, 
                                       b, ldb, beta, c, ldc);
         #endif
         break;
      case 'g': // gcn
         if (KRUNTIME_GCN && k >= BESTK_GCN) /* assumption: k >= BESTK */
            kk = BESTK_GCN/GVLEN; /* GVLEN: generated kernels vlen */
         else
         {
            kk = k / GVLEN;
            if (k % GVLEN || k > MAXDIM_SPMM) /* no optimize kernel */
            {
               trusted_fusedMM_gcn_csr(tkern, m, n, k, alpha, nnz, rows,
                                        cols, val, indx, pntrb, pntre, a, lda, 
                                        b, ldb, beta, c,ldc);
               return;
            }
         }
         if (beta == 0)
         #ifdef DREAL 
            dgenkernels_gcn_b0[kk-1](tkern, m, n, k, alpha, nnz, rows,
                                       cols, val, indx, pntrb, pntre, a, lda, 
                                       b, ldb, beta, c, ldc);
         #else
            sgenkernels_gcn_b0[kk-1](tkern, m, n, k, alpha, nnz, rows,
                                       cols, val, indx, pntrb, pntre, a, lda, 
                                       b, ldb, beta, c, ldc);
         #endif
         else /* beta == 1 */
         #ifdef DREAL 
            dgenkernels_gcn_b1[kk-1](tkern, m, n, k, alpha, nnz, rows,
                                       cols, val, indx, pntrb, pntre, a, lda, 
                                       b, ldb, beta, c, ldc);
         #else
            sgenkernels_gcn_b1[kk-1](tkern, m, n, k, alpha, nnz, rows,
                                       cols, val, indx, pntrb, pntre, a, lda, 
                                       b, ldb, beta, c, ldc);
         #endif
         break;
      default: 
         fprintf(stderr, "Kernel not implemented yet!!!\n");
         break;
   }
}

#ifdef __cplusplus
   }
#endif
