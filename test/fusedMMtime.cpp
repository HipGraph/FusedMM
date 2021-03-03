#include <cstdio>
#include <cstdint>
#include <random>
#include <cassert>
#include <omp.h>
#include "include/CSC.h"
#include "include/CSR.h"
#include "include/commonutility.h"
#include "include/utility.h"

/*
 * select data type in Makefile
 *    pre=[s,d]
 *    ityp=[int64_t,int32_t]
 */

#ifdef DREAL
   #define VALUETYPE double
#else
   #define VALUETYPE float
#endif

/*
 * Added header file for general fusedMM 
 */

#include "../fusedMM.h"


#ifdef INT64
   #ifndef INT64_MAX 
      #error "64bit integer not supported in this architecture!!!"
   #endif
#endif
#ifdef INT32
   #ifndef INT32_MAX 
      #error "32bit integer not supported in this architecture!!!"
   #endif
#endif

/*
 * some misc definition for timer : from ATLAS 
 */
#define ATL_MaxMalloc 268435456UL
#define ATL_Cachelen 64
   #define ATL_MulByCachelen(N_) ( (N_) << 6 )
   #define ATL_DivByCachelen(N_) ( (N_) >> 6 )

#define ATL_AlignPtr(vp) (void*) \
        ATL_MulByCachelen(ATL_DivByCachelen((((size_t)(vp))+ATL_Cachelen-1)))

/*
 * ===========================================================================
 * Defining API for our new kernel 
 * input: 
 *    Dense matrices: A -> MxK B -> NxK C -> MxD 
 *    Sparse: S -> MxN 
 * output:
 *    C -> MxK 
 *
 *    dot / sum /subtraction / t-dist
 *       - dot : scalar, tdist 
 *       - sum / subtraction : vector  
 *    sigmoid / scal 
 *
 *
 *
 *    Meta descriptor: 
 *      
 *
 * ============================================================================
 */

/* based on CSR */
typedef void (*csr_mm_t) 
(
   const char tkern,  // 'N' 'T'
   const INDEXTYPE m, 
   const INDEXTYPE n, 
   const INDEXTYPE k, 
   const VALUETYPE alpha, 
   const INDEXTYPE nnz,   // nonzeros: need to recreate csr with mkl 
   const INDEXTYPE rows,  // number of rows... not needed 
   const INDEXTYPE cols,  // number of columns 
   const VALUETYPE *val,   // NNZ value  
   const INDEXTYPE *indx,  // colids -> column indices 
   const INDEXTYPE *pntrb, // starting index for rowptr
   const INDEXTYPE *pntre, // ending index for rowptr
   const VALUETYPE *a,     // Dense B matrix
   const INDEXTYPE lda,   // 2nd dimension of b (col size since row-major)  
   const VALUETYPE *b,     // Dense B matrix
   const INDEXTYPE ldb,   // 2nd dimension of b (col size since row-major)  
   const VALUETYPE beta,  // beta value 
   VALUETYPE *c,           // Dense matrix c
   const INDEXTYPE ldc    // 2nd dimension size of c (col size since roa-major) 
);


/* based on CSC */
typedef void (*csc_mm_t) 
(
   const char tkern,  // 'N' 'T'
   const INDEXTYPE m, 
   const INDEXTYPE n, 
   const INDEXTYPE k, 
   const VALUETYPE alpha, 
   const INDEXTYPE nnz,   // nonzeros: need to recreate csr with mkl 
   const INDEXTYPE rows,  // number of rows... not needed 
   const INDEXTYPE cols,  // number of columns 
   const VALUETYPE *val,   // NNZ value  
   const INDEXTYPE *indx,  // colids -> column indices 
   const INDEXTYPE *pntrb, // starting index for rowptr
   const INDEXTYPE *pntre, // ending index for rowptr
   const VALUETYPE *a,     // Dense B matrix
   const INDEXTYPE lda,   // 2nd dimension of b (col size since row-major)  
   const VALUETYPE *b,     // Dense B matrix
   const INDEXTYPE ldb,   // 2nd dimension of b (col size since row-major)  
   const VALUETYPE beta,  // beta value 
   VALUETYPE *c,           // Dense matrix c
   const INDEXTYPE ldc    // 2nd dimension size of c (col size since roa-major) 
);

/* ============================================================================
 *    sample trusted kernels 
 *    Trusted kernels from gl/src/array/cpu/sddmmspmm 
 * 
 *============================================================================*/

#define SM_TABLE_SIZE 2048
#define SM_BOUND 5.0
#define SM_RESOLUTION SM_TABLE_SIZE/(2.0 * SM_BOUND)

template <typename DType>
DType scale(DType v)
{
   if(v > SM_BOUND) return SM_BOUND;
   else if(v < -SM_BOUND) return -SM_BOUND;
   return v;
}

template <typename DType>
DType fast_SM(DType v, DType *sm_table)
{
   if(v > SM_BOUND) return 1.0;
   else if(v < -SM_BOUND) return 0.0;
   return sm_table[(int)((v + SM_BOUND) * SM_RESOLUTION)];
}

template <typename IdType, typename DType>
void init_SM_TABLE(DType *sm_table)
{
   DType x;
   for(IdType i = 0; i < SM_TABLE_SIZE; i++)
   {
      x = 2.0 * SM_BOUND * i / SM_TABLE_SIZE - SM_BOUND;
      sm_table[i] = 1.0 / (1 + exp(-x));
   }
}

template <typename IdType, typename DType>
void SDDMMSPMMCsrTdist
(  const IdType *indptr, 
   const IdType *indices, 
   const IdType *edges,
   const DType *X, 
   const DType *Y, 
   DType *O, 
   const IdType N, 
   const int64_t dim) 
{

#ifdef PTTIME 
#pragma omp parallel for
#endif
   for (IdType rid = 0; rid < N; ++rid) 
   {
      const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
      const IdType iindex = rid * dim;
      DType T[dim];
      for (IdType j = row_start; j < row_end; ++j)
      {
         const IdType cid = indices[j];
         const IdType jindex = cid * dim;
         DType attrc = 0;
         for (int64_t k = 0; k < dim; ++k) 
         {
            T[k] = X[iindex + k] - Y[jindex + k];
            attrc += T[k] * T[k];
         }
      #if 0
         DType d1 = -2.0 / (1.0 + attrc);
         for (int64_t k = 0; k < dim; ++k) 
         {
            T[k] = scale<DType>(T[k] * d1);
            O[iindex+k] = O[iindex+k]  + T[k];
         }
      #else
         DType d1 = scale<DType>(-2.0 / (1.0 + attrc));
         for (int64_t k = 0; k < dim; ++k) 
         {
            T[k] = T[k] * d1;
            O[iindex+k] = O[iindex+k]  + T[k];
         }
      #endif
      }
   }
}

template <typename IdType, typename DType>
void SDDMMSPMMCsrSigmoid
(
   const IdType *indptr, 
   const IdType *indices, 
   const IdType *edges, 
   const DType *X, 
   const DType *Y, 
   DType *O, 
   const IdType N, 
   const int64_t dim
   ) 
{

   DType *sm_table;
   sm_table = static_cast<DType *> (::operator new (sizeof(DType[SM_TABLE_SIZE])));
   init_SM_TABLE<IdType, DType>(sm_table);
   //for(IdType i = 0; i < SM_TABLE_SIZE; i++) cout << sm_table[i] << " "; cout << endl;
#ifdef PTTIME 
#pragma omp parallel for
#endif
   for (IdType rid = 0; rid < N; ++rid)
   {
      const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
      const IdType iindex = rid * dim;
      for (IdType j = row_start; j < row_end; ++j)
      {
         const IdType cid = indices[j];
         const IdType jindex = cid * dim;
         DType attrc = 0;
         for (int64_t k = 0; k < dim; ++k) 
         {
            attrc += X[iindex + k] * Y[jindex + k];
         }
         //DType d1 = 1.0 / (1.0 + exp(-attrc));
         DType d1 = fast_SM<DType>(attrc, sm_table);
	 //printf("");
	 for (int64_t k = 0; k < dim; ++k) 
         {
            O[iindex+k] = O[iindex+k]  + (1.0 - d1) * Y[jindex + k];
         }
      }
   }		
}

template <typename IdType, typename DType>
void TrustedFR
(  const IdType *indptr,
   const IdType *indices,
   const IdType *edges,
   const DType *X,
   const DType *Y,
   DType *O,
   const IdType N,
   const int64_t dim)
{

#ifdef PTTIME 
#pragma omp parallel for
#endif
   for (IdType rid = 0; rid < N; ++rid) 
   {
      const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
      const IdType iindex = rid * dim;
      DType T[dim];
      for (IdType j = row_start; j < row_end; ++j)
      {
         const IdType cid = indices[j];
         const IdType jindex = cid * dim;
         DType attrc = 0;
         for (int64_t k = 0; k < dim; ++k) 
         {
            //T[k] = Y[jindex + k] - X[iindex + k];
            T[k] = X[iindex + k] - Y[jindex + k];  // need to verify 
            attrc += T[k] * T[k];
         }
         DType d1 = 1.0 + 1.0 / attrc;  // NOTE: do we need to mult by -1?  
         for (int64_t k = 0; k < dim; ++k) 
         {
            O[iindex+k] = O[iindex+k]  + d1 * T[k];
         }
      }
   }
}

void truested_spmm_csr 
(
   const char tkern,  // 'N' 'T'
   const INDEXTYPE m, 
   const INDEXTYPE n, 
   const INDEXTYPE k, 
   const VALUETYPE alpha, 
   const INDEXTYPE nnz,   // nonzeros: need to recreate csr with mkl 
   const INDEXTYPE rows,  // number of rows... not needed 
   const INDEXTYPE cols,  // number of columns 
   const VALUETYPE *val,   // NNZ value  
   const INDEXTYPE *indx,  // colids -> column indices 
   const INDEXTYPE *pntrb, // starting index for rowptr
   const INDEXTYPE *pntre, // ending index for rowptr
   const VALUETYPE *a,     // Dense B matrix
   const INDEXTYPE lda,   // 2nd dimension of b (col size since row-major)  
   const VALUETYPE *b,     // Dense B matrix
   const INDEXTYPE ldb,   // 2nd dimension of b (col size since row-major)  
   const VALUETYPE beta,  // beta value 
   VALUETYPE *c,           // Dense matrix c
   const INDEXTYPE ldc    // 2nd dimension size of c (col size since roa-major) 
)
{
#ifdef PTTIME 
   #pragma omp parallel for
#endif
   // spmm    
   for (INDEXTYPE i = 0; i < m; i++)
   {
      for (INDEXTYPE j=pntrb[i]; j < pntre[i]; j++)
      {
         for (INDEXTYPE kk=0; kk < k; kk++)
            c[i*ldc+kk] += (val[j]*b[indx[j]*ldb+kk]);
      }
   }

}

void truested_gcn_csr 
(
   const char tkern,  // 'N' 'T'
   const INDEXTYPE m, 
   const INDEXTYPE n, 
   const INDEXTYPE k, 
   const VALUETYPE alpha, 
   const INDEXTYPE nnz,   // nonzeros: need to recreate csr with mkl 
   const INDEXTYPE rows,  // number of rows... not needed 
   const INDEXTYPE cols,  // number of columns 
   const VALUETYPE *val,   // NNZ value  
   const INDEXTYPE *indx,  // colids -> column indices 
   const INDEXTYPE *pntrb, // starting index for rowptr
   const INDEXTYPE *pntre, // ending index for rowptr
   const VALUETYPE *a,     // Dense B matrix
   const INDEXTYPE lda,   // 2nd dimension of b (col size since row-major)  
   const VALUETYPE *b,     // Dense B matrix
   const INDEXTYPE ldb,   // 2nd dimension of b (col size since row-major)  
   const VALUETYPE beta,  // beta value 
   VALUETYPE *c,           // Dense matrix c
   const INDEXTYPE ldc    // 2nd dimension size of c (col size since roa-major) 
)
{
#ifdef PTTIME 
   #pragma omp parallel for
#endif
   // gcn   
   for (INDEXTYPE i = 0; i < m; i++)
   {
      for (INDEXTYPE j=pntrb[i]; j < pntre[i]; j++)
      {
         for (INDEXTYPE kk=0; kk < k; kk++)
            c[i*ldc+kk] += b[indx[j]*ldb+kk];
      }
   }
}
/*===========================================================================
 *    Trusted kernels from MKL, works only for SPMM  
 * 
 *===========================================================================*/

#ifdef TIME_MKL    /* defined to use MKL as trusted kerenl */

#include "mkl.h"
#include "mkl_spblas.h"
#include "mkl_types.h"

void MKL_csr_mm
(
   const char transa,     // 'N', 'T' , 'C' 
   const INDEXTYPE m,     // number of rows of A needed to compute 
   const INDEXTYPE n,     // number of cols of C
   const INDEXTYPE k,     // number of cols of A
   const VALUETYPE alpha, // double scalar ?? why ptr 
   const char *matdescra, // 6 characr array descriptor for A:
                          // [G/S/H/T/A/D],[L/U],[N/U],[F/C] -> [G,X,X,C] 
   const INDEXTYPE nnz,   // nonzeros: need to recreate csr with mkl 
   const INDEXTYPE rows,  // number of rows
   const INDEXTYPE cols,  // number of columns 
   const VALUETYPE *val,  // NNZ value  
   const INDEXTYPE *indx, // colids -> column indices 
   const INDEXTYPE *pntrb,// starting index for rowptr
   const INDEXTYPE *pntre,// ending index for rowptr
   const VALUETYPE *b,    // Dense B matrix
   const INDEXTYPE ldb,   // leading dimension of b for zero-based indexing  
   const VALUETYPE beta,  // double scalar beta[0] 
   VALUETYPE *c,          // Dense matrix c
   const INDEXTYPE ldc    // leading dimension size of c 
)
{
   INDEXTYPE i; 
   sparse_status_t stat; 
   sparse_matrix_t A = NULL; 
   struct matrix_descr Adsc; 

   // 1. inspection stage
/*
   sparse_status_t mkl_sparse_d_create_csr 
   (
      sparse_matrix_t *A, 
      const sparse_index_base_t indexing, 
      const MKL_INT rows, 
      const MKL_INT cols, 
      MKL_INT *rows_start,  // not const !!
      MKL_INT *rows_end,    // not const !!
      MKL_INT *col_indx,    // not const !!
      double *values        // not const
   );
   
   NOTE: NOTE: 
   -----------
   create_csr will overwrote rows_start, rows_end, col_indx and values
   So, we need to copy those here  
*/
   // copying CSR data, we will skip this copy in timing  
   MKL_INT M; 
   MKL_INT *rowptr;
   MKL_INT *col_indx;
   VALUETYPE *values; 

   // want to keep only one array for rowptr 
   //rowptr = (MKL_INT*) malloc((rows+1)*sizeof(MKL_INT));
/*
 * NOTE: we are allocating memory for full size. However, we can call the 
 * inspector and executor with partial/blocked 
 */
   M = m; 
   
   rowptr = (MKL_INT*) malloc((M+1)*sizeof(MKL_INT)); // just allocate upto M
   assert(rowptr);
   for (i=0; i < M; i++)
      rowptr[i] = pntrb[i];
   rowptr[i] = pntre[i-1];
   
   col_indx = (MKL_INT*) malloc(nnz*sizeof(MKL_INT));
   assert(col_indx);
   for (i=0; i < nnz; i++)
      col_indx[i] = indx[i]; 
   
   values = (VALUETYPE*) malloc(nnz*sizeof(VALUETYPE));
   assert(col_indx);
   for (i=0; i < nnz; i++)
      values[i] = val[i]; 

   cout << "--- Running inspector for MKL" << endl;

#ifdef DREAL 
   stat = mkl_sparse_d_create_csr (&A, SPARSE_INDEX_BASE_ZERO, M, cols, 
            rowptr, rowptr+1, col_indx, values);  
#else
   stat = mkl_sparse_s_create_csr (&A, SPARSE_INDEX_BASE_ZERO, M, cols, 
            rowptr, rowptr+1, col_indx, values);  
#endif

   if (stat != SPARSE_STATUS_SUCCESS)
   {
      cout << "creating csr for MKL failed!";
      exit(1);
   }
   // 2. execution stage 
/*
   sparse_status_t mkl_sparse_d_mm 
   (
      const sparse_operation_t operation, 
      const double alpha, 
      const sparse_matrix_t A, 
      const struct matrix_descr descr, 
      const sparse_layout_t layout, 
      const double *B, 
      const MKL_INT columns, 
      const MKL_INT ldb, 
      const double beta, 
      double *C, 
      const MKL_INT ldc);
*/ 
   Adsc.type = SPARSE_MATRIX_TYPE_GENERAL;
   //Adsc.fill // no need for general matrix  
   Adsc.diag =  SPARSE_DIAG_NON_UNIT;  // no need for general 
   
   cout << "--- Running executor for MKL" << endl;
#ifdef DREAL 
   stat = mkl_sparse_d_mm (SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, Adsc, 
                           SPARSE_LAYOUT_ROW_MAJOR, b, n, ldb, beta, c, ldc); 
#else
   stat = mkl_sparse_s_mm (SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, Adsc, 
                           SPARSE_LAYOUT_ROW_MAJOR, b, n, ldb, beta, c, ldc); 
#endif

   if (stat != SPARSE_STATUS_SUCCESS)
   {
      cout << "creating csr for MKL failed!";
      exit(1);
   }
   cout << "--- Done calling MKL's API" << endl;
/*
 * free all data 
 */
   free(rowptr);
   free(col_indx);
   free(values);
   mkl_sparse_destroy(A);
}
#endif  /* END TIME_MKL */

void mytrusted_csr 
(
   const char tkern,  // 'N' 'T'
   const INDEXTYPE m, 
   const INDEXTYPE n, 
   const INDEXTYPE k, 
   const VALUETYPE alpha, 
   const INDEXTYPE nnz,   // nonzeros: need to recreate csr with mkl 
   const INDEXTYPE rows,  // number of rows... not needed 
   const INDEXTYPE cols,  // number of columns 
   const VALUETYPE *val,   // NNZ value  
   const INDEXTYPE *indx,  // colids -> column indices 
   const INDEXTYPE *pntrb, // starting index for rowptr
   const INDEXTYPE *pntre, // ending index for rowptr
   const VALUETYPE *a,     // Dense B matrix
   const INDEXTYPE lda,   // 2nd dimension of b (col size since row-major)  
   const VALUETYPE *b,     // Dense B matrix
   const INDEXTYPE ldb,   // 2nd dimension of b (col size since row-major)  
   const VALUETYPE beta,  // beta value 
   VALUETYPE *c,           // Dense matrix c
   const INDEXTYPE ldc    // 2nd dimension size of c (col size since roa-major) 
)
{
   switch(tkern)
   {
#ifndef TIME_MKL
      case 't' : // t-dist 
         //fprintf(stderr, "***Applying trusted t-dist kernel: (m,k) = %d, %d\n", m, k);
         SDDMMSPMMCsrTdist<INDEXTYPE, VALUETYPE> (pntrb, indx, NULL, a, b, c, 
               m, k);
         break;
      case 's' : // sigmoid
         //fprintf(stderr, "***Applying trusted sigmoid kernel\n");
         SDDMMSPMMCsrSigmoid<INDEXTYPE, VALUETYPE> (pntrb, indx, NULL, a, b, c, 
               m, k);
         break;
      case 'm' : // spmm
         //fprintf(stderr, "***Applying trusted spmm kernel\n");
         truested_spmm_csr(tkern, m, n, k, alpha, nnz, rows, cols, val, indx, 
               pntrb, pntre, a, lda, b, ldb, beta, c, ldc);
         break;
      case 'g' : // gcn
         //fprintf(stderr, "***Applying trusted gcn kernel\n");
         truested_gcn_csr(tkern, m, n, k, alpha, nnz, rows, cols, val, indx, 
               pntrb, pntre, a, lda, b, ldb, beta, c, ldc);
         break;
      case 'f' :
	 TrustedFR<INDEXTYPE, VALUETYPE> (pntrb, indx, NULL, a, b, c, m, k);
	 break;
#else  /* MKL only supports SPMM */
      case 'm' : // MKL spmm
         // no 'a' matrix, only: S(sparse), B, C
         // NOTE: NOYE: S -> mxn, b -> nxk, c-> mxk meaning MKL's n <-> k
         MKL_csr_mm('N', m, k, n, alpha, "GXXC", nnz, rows, cols, val, indx, 
               pntrb, pntre, b, ldb, beta, c, ldc);
         break;
#endif
      default:
         printf("unknown trusted kernel, timing is exiting... ... ...\n");
         exit(1);
   }
}

/*=============================================================================
 * Test kernels: 
 *    We will always call fusedMM, analyzing patterns it may call optimized 
 *    kernel from there
 *
 *============================================================================*/
/* **********************************************************************
 * Accessory funciton to compute sigmoid 
 */
/* scalar scale function */
VALUETYPE *SM_TABLE;
inline VALUETYPE uscale_SM(VALUETYPE val)
{
   VALUETYPE sval;
   /* hopefully compiler will figure out and replace it max min instruction */
   sval = (val > SM_BOUND) ? SM_BOUND : val;
   sval = (val < -SM_BOUND) ? -SM_BOUND : val;
   return(sval); 
}
/* not even parallel ?? */
void uinit_SM_TABLE()
{
   VALUETYPE x;
   SM_TABLE = (VALUETYPE*)malloc(SM_TABLE_SIZE*sizeof(VALUETYPE));
   assert(SM_TABLE);
   for(INDEXTYPE i = 0; i < SM_TABLE_SIZE; i++)
   {
      x = 2.0 * SM_BOUND * i / SM_TABLE_SIZE - SM_BOUND;
      SM_TABLE[i] = 1.0 / (1 + exp(-x));
   }
}

VALUETYPE ufast_SM(VALUETYPE v)
{
   if (v > SM_BOUND) return 1.0;
   else if (v < -SM_BOUND) return 0.0;
   return SM_TABLE[(INDEXTYPE)((v + SM_BOUND) * SM_RESOLUTION)];
}

VALUETYPE tscale(VALUETYPE v)
{
   if(v > SM_BOUND) return SM_BOUND;
   else if(v < -SM_BOUND) return -SM_BOUND;
   return v;
}

/*
 * NOTE: implementation of User defined functions differ from different model.
 * We need to enable disable it compile time!!!!
 * FIXME: How to select them runtime 
 */

extern "C" int SOP_UDEF_FUNC(VALUETYPE val, VALUETYPE *out);
#ifdef SIGMOID_UDEF 
// USER DEFINED FUNCTION for SOP with Sigmoid calc 
int  SOP_UDEF_FUNC(VALUETYPE val, VALUETYPE *out)
{
   *out = 1.0 - ufast_SM(val);
   return FUSEDMM_SUCCESS_RETURN;
}
#elif defined(FR_UDEF)
// SOP_UDEF for FR model
int SOP_UDEF_FUNC(VALUETYPE val, VALUETYPE *out)
{
   *out = 1.0 + 1.0 / val;
   return FUSEDMM_SUCCESS_RETURN;
}
#elif defined(TDIST_UDEF)
// SOP_UDEF for t-distribution  model
int SOP_UDEF_FUNC(VALUETYPE val, VALUETYPE *out)
{  
   *out = tscale(-2.0 / (1.0 + val));
   return FUSEDMM_SUCCESS_RETURN;
}
#elif defined(LL_UDEF)
// SOP_UDEF for LL model
int SOP_UDEF_FUNC(VALUETYPE val, VALUETYPE *out)
{
   *out = log2(1 + sqrt(val));;
   return FUSEDMM_SUCCESS_RETURN;
}
#elif defined(FA_UDEF)
// SOP_UDEF for FA model
int SOP_UDEF_FUNC(VALUETYPE val, VALUETYPE *out)
{
   *out = sqrt(val) + 1.0 / val;;
   return FUSEDMM_SUCCESS_RETURN;
} 
#else 
/*
 * NOTE: other kernels don't use SOP funciton (NOOP or COPY)
 * However, since we enable SOP_UDEF_IMPL in fusedMM.h, we need a dummy func.
 * Normally, users should disable the macro if they don't want to provide any 
 * implementation. We are using this dummy since we use same source for all 
 * the different executables.
 * NOTE: don't use _gcn to run sigmoid kernel
 */
int SOP_UDEF_FUNC(VALUETYPE val, VALUETYPE *out)
{
   *out = val;;
   return FUSEDMM_SUCCESS_RETURN;
} 
#endif
#if 0
/*
 * User defined function for ROP to perform self-DOT product
 */
// tdist 
int ROP_UDEF_FUNC(INDEXTYPE lhs_dim, const VALUETYPE *lhs, INDEXTYPE rhs_dim,
      const VALUETYPE *rhs, VALUETYPE &out)
{
   out = 0.0;
   for (INDEXTYPE i = 0; i < rhs_dim; i += 1)
   {
      out += rhs[i] * rhs[i];
   }  
   return FUSEDMM_SUCCESS_RETURN;
}
// tdist 
int VSC_UDEF_FUNC(INDEXTYPE rhs_dim, const VALUETYPE *rhs, VALUETYPE scal,
      INDEXTYPE out_dim, VALUETYPE *out)
{
   for (INDEXTYPE i = 0; i < rhs_dim; i += 1)
   {
      out[i] = scale(scal * rhs[i]);
   }
   return FUSEDMM_SUCCESS_RETURN;
}
#endif

   
void mytest_csr
(
   const char tkern,  // 'N' 'T'
   const INDEXTYPE m, 
   const INDEXTYPE n, 
   const INDEXTYPE k, 
   const VALUETYPE alpha, 
   const INDEXTYPE nnz,   // nonzeros: need to recreate csr with mkl 
   const INDEXTYPE rows,  // number of rows... not needed 
   const INDEXTYPE cols,  // number of columns 
   const VALUETYPE *val,   // NNZ value  
   const INDEXTYPE *indx,  // colids -> column indices 
   const INDEXTYPE *pntrb, // starting index for rowptr
   const INDEXTYPE *pntre, // ending index for rowptr
   const VALUETYPE *a,     // Dense B matrix
   const INDEXTYPE lda,   // 2nd dimension of b (col size since row-major)  
   const VALUETYPE *b,     // Dense B matrix
   const INDEXTYPE ldb,   // 2nd dimension of b (col size since row-major)  
   const VALUETYPE beta,  // beta value 
   VALUETYPE *c,           // Dense matrix c
   const INDEXTYPE ldc    // 2nd dimension size of c (col size since roa-major) 
)
{
   int32_t imsg; 
   switch(tkern)
   {
      case 't' : // t-dist 
	 imsg = VOP_SUBR | ROP_NORMR | SOP_UDEF | VSC_MUL | AOP_ADD;
	 fusedMM_csr(imsg, m, n, k, alpha, nnz, rows, cols, val, indx, pntrb,
               pntre, a, lda, b, ldb, beta, c, ldc);
         break;
      case 'f':
         //printf("Calling FR model!");
	 imsg = VOP_SUBR | ROP_NORMR | SOP_UDEF | VSC_MUL | AOP_ADD;
	 fusedMM_csr(imsg, m, n, k, alpha, nnz, rows, cols, val, indx, pntrb,
               pntre, a, lda, b, ldb, beta, c, ldc);
	 break;
      case 's' : // sigmoid
         uinit_SM_TABLE();
         //printf("Calling fused kernel\n");
         //imsg = VOP_COPY_LHS | ROP_DOT | SOP_UDEF | VSC_MUL | AOP_ADD;
         imsg = VOP_COPY_RHS | ROP_DOT | SOP_UDEF | VSC_MUL | AOP_ADD;
         
         fusedMM_csr(imsg, m, n, k, alpha, nnz, rows, cols, val, indx, pntrb, 
               pntre, a, lda, b, ldb, beta, c, ldc);
         
         break;
      case 'm' : // spmm
         imsg = VOP_COPY_RHS | ROP_NOOP | SOP_COPY | VSC_MUL | AOP_ADD;
         
         fusedMM_csr(imsg, m, n, k, alpha, nnz, rows, cols, val, indx, pntrb, 
               pntre, a, lda, b, ldb, beta, c, ldc);
         break;
      case 'g' : // gcn 
         imsg = VOP_COPY_RHS | ROP_NOOP | SOP_NOOP | VSC_NOOP | AOP_ADD;
         
         fusedMM_csr(imsg, m, n, k, alpha, nnz, rows, cols, val, indx, pntrb, 
               pntre, a, lda, b, ldb, beta, c, ldc);
         break;
      default:
         printf("unknown trusted kernel\n");
         break;
   }
}


/*=============================================================================
 *          Tester framework 
 * We will redesign it with tester class later... just using template here
 * ============================================================================
 */
// from ATLAS; ATL_epsilon.c 
template <typename NT> 
NT Epsilon(void)
{
   static NT eps; 
   const NT half=0.5; 
   volatile NT maxval, f1=0.5; 

   do
   {
      eps = f1;
      f1 *= half;
      maxval = 1.0 + f1;
   }
   while(maxval != 1.0);
   return(eps);
}

template <typename IT, typename NT>
int doChecking(IT NNZA, IT M, IT N, NT *C, NT *D, IT ldc)
{
   IT i, j, k;
   NT diff, EPS; 
   double ErrBound; 

   int nerr = 0;
/*
 * Error bound : total computation = K*NNZ + K*NNZ FMAC = 4*K*NNZ
 *               flop per element of C = 4*K*NNZ / M*K
 *
 */
   EPS = Epsilon<NT>();
   // the idea is how many flop one element needs, should be max degree
   // NOTE: avg degree will not do, since some rows may have more non-zero 
   // 2 for opposit direction of errors 
   ErrBound = 2 * 4 * (NNZA) * EPS; /* considering upper bound for now*/ 
   //cout << "--- EPS = " << EPS << " ErrBound = " << ErrBound << endl; 
   //cout << "--- ErrBound = " << ErrBound << " NNZ(A) = " << NNZA << " N = " << N  <<endl; 
   // row major! 
   for (i=0; i < M; i++)
   {
      for (j=0; j < N; j++)
      {
         k = i*ldc + j;
         diff = C[k] - D[k];
         if (diff < 0.0) diff = -diff; 
         if (diff > ErrBound)
         {
      #if 0
            fprintf(stderr, "C(%d,%d) : expected=%e, got=%e, diff=%e\n",
                    i, j, C[k], D[k], diff);
      #else // print single value... 
            if (!i && !j)
               fprintf(stderr, "C(%ld,%ld) : expected=%e, got=%e, diff=%e\n",
                       i, j, C[k], D[k], diff);
      #endif
            nerr++;
         }
         else if (D[k] != D[k]) /* test for NaNs */
         {
            fprintf(stderr, "C(%ld,%ld) : expected=%e, got=%e\n",
                    i, j, C[k], D[k]);
            nerr++;

         }
      }
   }
   return(nerr);
}

template <csr_mm_t trusted, csr_mm_t test>
int doTesting_Acsr
(
   CSR<INDEXTYPE,VALUETYPE> &S, 
   INDEXTYPE M, 
   INDEXTYPE N, 
   INDEXTYPE K, 
   VALUETYPE alpha, 
   VALUETYPE beta,
   int tkern
)
{
   int nerr, szAligned; 
   size_t i, j, szA, szB, szC, lda, ldc, ldb; 
   VALUETYPE *pb, *b, *pc0, *c0, *pc, *c, *pa, *a, *values;

   std::default_random_engine generator;
   std::uniform_real_distribution<VALUETYPE> distribution(0.0,1.0);
/*
 * NOTE: we are considering only row major A, B and C storage now
 *       A -> MxK, B->NxK, C->MxD  
 */
   lda = ldb = ldc = K; // both row major, K multiple of VLEN 
/*
 * NOTE: not sure about system's VLEN from this user code. So, make it cacheline
 * size aligned ....
 */
   szAligned = ATL_Cachelen / sizeof(VALUETYPE);
   szA = ((M*ldb+szAligned-1)/szAligned)*szAligned;  // szB in element
   szB = ((N*ldb+szAligned-1)/szAligned)*szAligned;  // szB in element
   szC = ((M*ldc+szAligned-1)/szAligned)*szAligned;  // szC in element 
   
   pa = (VALUETYPE*)malloc(szA*sizeof(VALUETYPE)+2*ATL_Cachelen);
   assert(pa);
   a = (VALUETYPE*) ATL_AlignPtr(pa);
   
   pb = (VALUETYPE*)malloc(szB*sizeof(VALUETYPE)+2*ATL_Cachelen);
   assert(pb);
   b = (VALUETYPE*) ATL_AlignPtr(pb);

   pc0 = (VALUETYPE*)malloc(szC*sizeof(VALUETYPE)+2*ATL_Cachelen);
   assert(pc0);
   c0 = (VALUETYPE*) ATL_AlignPtr(pc0); 
      
   pc = (VALUETYPE*)malloc(szC*sizeof(VALUETYPE)+2*ATL_Cachelen);
   assert(pc);
   c = (VALUETYPE*) ATL_AlignPtr(pc); 
   
   // init   
   for (i=0; i < szA; i++)
   {
   #if 1
      a[i] = distribution(generator);  
   #else
      //a[i] = 1.0*i;  
      a[i] = 0.5;  
   #endif
   }
   for (i=0; i < szB; i++)
   {
   #if 1
      b[i] = distribution(generator);  
   #else
      //b[i] = 1.0*i;  
      b[i] = 0.5;  
   #endif
   }
   for (i=0; i < szC; i++)
   {
   #if 0
      c[i] = c0[i] = distribution(generator);  
   #else  /* to test beta0 case */
      c[i] = 0.0; c0[i] = 0.0;
   #endif
   }
  
   if (M > S.rows) M = S.rows; // M can't be greater than A.rows  
/*
 *    csr may consists all 1 as values... init with random values
 */
   values = (VALUETYPE*)malloc(S.nnz*sizeof(VALUETYPE));
   assert(values);
   for (i=0; i < S.nnz; i++)
      values[i] = distribution(generator);  
/*
 * Let's apply trusted and test kernels 
 */
   fprintf(stdout, "Applying trusted kernel\n");
   //trusted(tkern, M, N, K, alpha, S.nnz, S.rows, S.cols, 
   //        S.colids, S.rowptr, S.rowptr+1, a, lda, b, ldb, beta, c0, ldc);   
   trusted(tkern, M, N, K, alpha, S.nnz, S.rows, S.cols, values, 
           S.colids, S.rowptr, S.rowptr+1, a, lda, b, ldb, beta, c0, ldc);   
   
   fprintf(stdout, "Applying test kernel\n");
   //test(tkern, M, N, K, alpha, S.nnz, S.rows, S.cols, 
   //      S.colids, S.rowptr, S.rowptr+1, a, lda, b, ldb, beta, c, ldc);   
   test(tkern, M, N, K, alpha, S.nnz, S.rows, S.cols, values, 
         S.colids, S.rowptr, S.rowptr+1, a, lda, b, ldb, beta, c, ldc);   
/*
 * check for errors 
 */
   nerr = doChecking<INDEXTYPE, VALUETYPE>(S.nnz, M, K, c0, c, ldc);

   free(values);
   free(pc0);
   free(pc);
   free(pb);
   free(pa);

   return(nerr);
}
/*==============================================================================
 *    Timer:  
 *
 *============================================================================*/
/*
 * NOTE: kernel timer prototype, typedef template function pointer   
 */
template <typename IT>
using csr_timer_t = vector<double> (*) 
(
   const int tkern,         // kernel type
   const int nrep,         // number of repeatation 
   const IT M,      
   const IT N,     
   const IT K,    
   const VALUETYPE alpha,  // alpha
   const IT nnz,
   const IT rows,
   const IT cols,
   VALUETYPE *values,      // values
   IT *rowptr,   
   IT *colids,
   const VALUETYPE *a,
   const IT lda,
   const VALUETYPE *b,
   const IT ldb,
   const VALUETYPE beta,
   VALUETYPE *c,
   const IT ldc
);
template <typename IT>
using csr_timer_cf_t = vector<double> (*) 
(
   const IT ndsets,
   const IT wdsz,
   const IT nisets,
   const IT wisz,
   const int nrep,         // number of repeatation 
   const int tkern,         // kernel type
   const IT M,      
   const IT N,     
   const IT K,    
   const VALUETYPE alpha,  // alpha
   const IT nnz,
   const IT rows,
   const IT cols,
   VALUETYPE *values,      // values
   IT *rowptr,   
   IT *colids,
   const VALUETYPE *a,
   const IT lda,
   const VALUETYPE *b,
   const IT ldb,
   const VALUETYPE beta,
   VALUETYPE *c,
   const IT ldc
);
/*
 * Kernel timer wrapper for trusted kernel.. 
 * This wrapper handles all extra setup needed to call a library, like: MKL
 */
vector<double> callTimerTrusted_Acsr
(
   const int tkern,      // ROW_MAJOR, INDEX_BASE_ZERO 
   const int nrep,      // number of repeatation 
   const INDEXTYPE M,
   const INDEXTYPE N,
   const INDEXTYPE K, // A.cols
   const VALUETYPE alpha,
   const INDEXTYPE nnz,
   const INDEXTYPE rows,
   const INDEXTYPE cols,
   VALUETYPE *values, 
   INDEXTYPE *rowptr,
   INDEXTYPE *colids,
   const VALUETYPE *a,     
   const INDEXTYPE lda,   
   const VALUETYPE *b,
   const INDEXTYPE ldb,
   const VALUETYPE beta,
   VALUETYPE *c,
   const INDEXTYPE ldc
)
{
   double start, end;
   vector <double> results;  // don't use single precision, use double  
/*
 * NOTE: 
 *    flag can be used to select different option, like: ROW_MAJOR, 
 *    INDEX_BASE_ZERO. For now, we only support following options (no checking):
 *       SPARSE_INDEX_BASE_ZERO
 *       SPARSE_OPERATION_NON_TRANSPOSE
 *       SPARSE_LAYOUT_ROW_MAJOR
 */
   // timing inspector phase 
   {
      results.push_back(0.0); // no inspection phase 
   }
   
   mytrusted_csr(tkern, M, N, K, alpha, nnz, rows, cols, values, 
              colids, rowptr, rowptr+1, a, lda, b, ldb, beta, c, ldc);   
   
   start = omp_get_wtime();
   for (int i=0; i < nrep; i++)
   {
      mytrusted_csr(tkern, M, N, K, alpha, nnz, rows, cols, values, 
                 colids, rowptr, rowptr+1, a, lda, b, ldb, beta, c, ldc);   
   }
   end = omp_get_wtime();
   results.push_back((end-start)/((double)nrep)); // execution time 

   return(results);
}

// cache blocking version 
vector<double> callCFTimerTrusted_Acsr
(
   const INDEXTYPE ndsets,
   const INDEXTYPE wdsz,
   const INDEXTYPE nisets,
   const INDEXTYPE wisz,
   const int nrep,      // number of repeatation 
   const int tkern,      // ROW_MAJOR, INDEX_BASE_ZERO 
   const INDEXTYPE M,
   const INDEXTYPE N,
   const INDEXTYPE K, // A.cols
   const VALUETYPE alpha,
   const INDEXTYPE nnz,
   const INDEXTYPE rows,
   const INDEXTYPE cols,
   VALUETYPE *values, 
   INDEXTYPE *rowptr,
   INDEXTYPE *colids,
   const VALUETYPE *a,     
   const INDEXTYPE lda,   
   const VALUETYPE *b,
   const INDEXTYPE ldb,
   const VALUETYPE beta,
   VALUETYPE *c,
   const INDEXTYPE ldc
)
{
   INDEXTYPE nds = ndsets;
   INDEXTYPE nis = nisets;
   double start, end;
   vector <double> results;  // don't use single precision, use double  
   
   // timing inspector phase 
   {
      results.push_back(0.0); // no inspection phase 
   }

#if 0   
   mytrusted_csr(tkern, M, N, K, alpha, nnz, rows, cols, values, 
              colids, rowptr, rowptr+1, a, lda, b, ldb, beta, c, ldc);   
#endif
   
   start = omp_get_wtime();
   for (int i=0; i < nrep; i++)
   {
      nds--; nis--; 
      mytrusted_csr(tkern, M, N, K, alpha, nnz, rows, cols, 
            values+nds*wdsz, colids+nis*wisz, rowptr+nis*wisz, 
            rowptr+nis*wisz+1, a+nds*wdsz, lda, b+nds*wdsz, ldb, beta, 
            c+nds*wdsz, ldc);   
      if (!nds)
         nds = ndsets;
      if (!nis)
         nis = nisets;
   }
   end = omp_get_wtime();
   results.push_back((end-start)/((double)nrep)); // execution time 

   return(results);
}

#if TIME_MKL 
/*
 * kernel timer wrapper for MKL
 * NOTE: MKL: M,N,K  => FUUSEDMM : M,K,N
 * always ROW_MAJOR, INDEX_BASE_ZERO  
 */
vector<double> callTimerMKL_Acsr
(
   const int tkern,      // always 'm'
   const int nrep,      // number of repeatation 
   const MKL_INT M,
   const MKL_INT N,
   const MKL_INT K, 
   const VALUETYPE alpha,
   const MKL_INT nnz,
   const MKL_INT rows,
   const MKL_INT cols,
   VALUETYPE *values, 
   MKL_INT *rowptr,
   MKL_INT *colids,
   const VALUETYPE *a,   /* not used in spmm */
   const MKL_INT lda,    /* not used in spmm */
   const VALUETYPE *b,
   const MKL_INT ldb,
   const VALUETYPE beta,
   VALUETYPE *c,
   const MKL_INT ldc
)
{
   double start, end;
   vector <double> results;  // don't use single precision, use double  
   MKL_INT n, k; 
   sparse_status_t stat; 
   sparse_matrix_t Amkl = NULL; 
   struct matrix_descr Adsc;

   n = K; k = N; /* difference in notation for FUSEDMM and MKL */
/*
 * NOTE: 
 *    flag can be used to select different option, like: ROW_MAJOR, 
 *    INDEX_BASE_ZERO. For now, we only support following options (no checking):
 *       SPARSE_INDEX_BASE_ZERO
 *       SPARSE_OPERATION_NON_TRANSPOSE
 *       SPARSE_LAYOUT_ROW_MAJOR
 */
   // timing inspector phase 
   {
/*
 *	force mkl to use specified threads 
 */
   #ifdef PTTME
      #ifdef NTHREADS
         //cout << "setting mkl threads = " << NTHREADS << endl;
         mkl_set_num_threads(NTHREADS); 
      #endif
   #endif

      start = omp_get_wtime();
   #ifdef DREAL 
      stat = mkl_sparse_d_create_csr (&Amkl, SPARSE_INDEX_BASE_ZERO, M, k, 
               rowptr, rowptr+1, colids, values);  
   #else
      stat = mkl_sparse_s_create_csr (&Amkl, SPARSE_INDEX_BASE_ZERO, M, k, 
               rowptr, rowptr+1, colids, values);  
   #endif
      end = omp_get_wtime();
      results.push_back(end-start); // setup time 
#if 0
      if (stat != SPARSE_STATUS_SUCCESS)
      {
         cout << "creating csr for MKL failed!";
         exit(1);
      }
#endif
   }
   
   Adsc.type = SPARSE_MATRIX_TYPE_GENERAL;
   //Adsc.fill // no need for general matrix  
   //Adsc.diag =  SPARSE_DIAG_NON_UNIT;  // no need for general 
  
   // skiping first call 
   #ifdef DREAL 
      stat = mkl_sparse_d_mm (SPARSE_OPERATION_NON_TRANSPOSE, alpha, Amkl, Adsc, 
                              SPARSE_LAYOUT_ROW_MAJOR, b, n, ldb, beta, c, ldc); 
   #else
      stat = mkl_sparse_s_mm (SPARSE_OPERATION_NON_TRANSPOSE, alpha, Amkl, Adsc, 
                              SPARSE_LAYOUT_ROW_MAJOR, b, n, ldb, beta, c, ldc); 
   #endif
#if 1
   if (stat != SPARSE_STATUS_SUCCESS)
   {
      cout << "creating csr for MKL failed, stat =!" << SPARSE_STATUS_SUCCESS 
           << endl;
      exit(1);
   }
#endif
   
   start = omp_get_wtime();
   for (int i=0; i < nrep; i++)
   {
   #ifdef DREAL 
      stat = mkl_sparse_d_mm (SPARSE_OPERATION_NON_TRANSPOSE, alpha, Amkl, Adsc, 
                              SPARSE_LAYOUT_ROW_MAJOR, b, n, ldb, beta, c, ldc); 
   #else
      stat = mkl_sparse_s_mm (SPARSE_OPERATION_NON_TRANSPOSE, alpha, Amkl, Adsc, 
                              SPARSE_LAYOUT_ROW_MAJOR, b, n, ldb, beta, c, ldc); 
   #endif
   }
   end = omp_get_wtime();
   results.push_back((end-start)/((double)nrep)); // execution time 

   mkl_sparse_destroy(Amkl);
   return(results);
}
#endif   /* END OF TIME_MKL */

/*
 * timer wrapper for test kernel 
 */
vector<double> callTimerTest_Acsr
(
   const int tkern,      // kernel type  
   const int nrep,      // number of repeatation 
   const INDEXTYPE M,
   const INDEXTYPE N,
   const INDEXTYPE K, 
   const VALUETYPE alpha,
   const INDEXTYPE nnz,
   const INDEXTYPE rows,
   const INDEXTYPE cols,
   VALUETYPE *values, 
   INDEXTYPE *rowptr,
   INDEXTYPE *colids,
   const VALUETYPE *a,     
   const INDEXTYPE lda,   
   const VALUETYPE *b,
   const INDEXTYPE ldb,
   const VALUETYPE beta,
   VALUETYPE *c,
   const INDEXTYPE ldc
)
{
   double start, end;
   vector <double> results;  // don't use single precision, use double  
/*
 * NOTE: 
 *    flag can be used to select different option, like: ROW_MAJOR, 
 *    INDEX_BASE_ZERO. For now, we only support following options (no checking):
 *       SPARSE_INDEX_BASE_ZERO
 *       SPARSE_OPERATION_NON_TRANSPOSE
 *       SPARSE_LAYOUT_ROW_MAJOR
 */
   // timing inspector phase 
   {
      results.push_back(0.0); // no inspection phase 
   }
#if 0  
#ifdef DREAL
   dgfusedMM_csr(tkern, M, N, K, alpha, nnz, rows, cols, values, 
              colids, rowptr, rowptr+1, a, lda, b, ldb, beta, c, ldc);   
#else
   sgfusedMM_csr(tkern, M, N, K, alpha, nnz, rows, cols, values, 
              colids, rowptr, rowptr+1, a, lda, b, ldb, beta, c, ldc);   
#endif
#else  // calling general fusedmm 
   mytest_csr(tkern, M, N, K, alpha, nnz, rows, cols, values, 
              colids, rowptr, rowptr+1, a, lda, b, ldb, beta, c, ldc);   
#endif
   start = omp_get_wtime();
   for (int i=0; i < nrep; i++)
   {
#if 0
   #ifdef DREAL
      dgfusedMM_csr(tkern, M, N, K, alpha, nnz, rows, cols, values, 
                 colids, rowptr, rowptr+1, a, lda, b, ldb, beta, c, ldc);   
   #else
      sgfusedMM_csr(tkern, M, N, K, alpha, nnz, rows, cols, values, 
                 colids, rowptr, rowptr+1, a, lda, b, ldb, beta, c, ldc);   
   #endif
#else  // calling general fusedmm 
      mytest_csr(tkern, M, N, K, alpha, nnz, rows, cols, values, 
                 colids, rowptr, rowptr+1, a, lda, b, ldb, beta, c, ldc);   
#endif
   }
   end = omp_get_wtime();
   results.push_back((end-start)/((double)nrep)); // execution time 

   return(results);
}

vector<double> callCFTimerTest_Acsr
(
   const INDEXTYPE ndsets,
   const INDEXTYPE wdsz,
   const INDEXTYPE nisets,
   const INDEXTYPE wisz,
   const int nrep,      // number of repeatation 
   const int tkern,      // kernel type  
   const INDEXTYPE M,
   const INDEXTYPE N,
   const INDEXTYPE K, 
   const VALUETYPE alpha,
   const INDEXTYPE nnz,
   const INDEXTYPE rows,
   const INDEXTYPE cols,
   VALUETYPE *values, 
   INDEXTYPE *rowptr,
   INDEXTYPE *colids,
   const VALUETYPE *a,     
   const INDEXTYPE lda,   
   const VALUETYPE *b,
   const INDEXTYPE ldb,
   const VALUETYPE beta,
   VALUETYPE *c,
   const INDEXTYPE ldc
)
{
   INDEXTYPE nds = ndsets;
   INDEXTYPE nis = nisets;
   double start, end;
   vector <double> results;  // don't use single precision, use double  
   // timing inspector phase 
   {
      results.push_back(0.0); // no inspection phase 
   }
   start = omp_get_wtime();
   for (int i=0; i < nrep; i++)
   {
      nds--; nis--; 
#if 0
   #ifdef DREAL
      dgfusedMM_csr(tkern, M, N, K, alpha, nnz, rows, cols, 
            values+nds*wdsz, colids+nis*wisz, rowptr+nis*wisz, 
            rowptr+nis*wisz+1, a+nds*wdsz, lda, b+nds*wdsz, ldb, beta, 
            c+nds*wdsz, ldc);   
   #else
      sgfusedMM_csr(tkern, M, N, K, alpha, nnz, rows, cols, 
            values+nds*wdsz, colids+nis*wisz, rowptr+nis*wisz, 
            rowptr+nis*wisz+1, a+nds*wdsz, lda, b+nds*wdsz, ldb, beta, 
            c+nds*wdsz, ldc);   
   #endif
#else  // calling general fusedmm 
      mytest_csr(tkern, M, N, K, alpha, nnz, rows, cols, 
            values+nds*wdsz, colids+nis*wisz, rowptr+nis*wisz, 
            rowptr+nis*wisz+1, a+nds*wdsz, lda, b+nds*wdsz, ldb, beta, 
            c+nds*wdsz, ldc);   
#endif
      if (!nds)
         nds = ndsets;
      if (!nis)
         nis = nisets;
   }
   end = omp_get_wtime();
   results.push_back((end-start)/((double)nrep)); // execution time 

   return(results);
}

/*
 * Assuming large working set, sizeof B+D > L3 cache 
 */
template<typename IT, csr_timer_t<IT> CSR_TIMER>
vector <double> doTiming_Acsr
(
 const CSR<INDEXTYPE, VALUETYPE> &S, 
 IT M, 
 IT N, 
 IT K,
 const VALUETYPE alpha,
 const VALUETYPE beta,
 const int csKB,
 const int nrep,
 const int tkern
 )
{
   int szAligned; 
   IT i, j;
   vector <double> results; 
   double start, end;
   IT nnz, rows, cols;
   //size_t szB, szC, ldb, ldc; 
   IT szA, szB, szC, lda, ldb, ldc; 
   VALUETYPE *pa, *a, *pb, *b, *pc, *c, *values;
   IT *rowptr, *colids;

#if defined(PTTIME) && defined(NTHREADS)
   omp_set_num_threads(NTHREADS);
#endif

   std::default_random_engine generator;
   std::uniform_real_distribution<double> distribution(0.0,1.0);

   lda = ldb = ldc = K; // considering both row-major   

   szAligned = ATL_Cachelen / sizeof(VALUETYPE);
   szA = ((M*ldb+szAligned-1)/szAligned)*szAligned;  // szB in element
   szB = ((N*ldb+szAligned-1)/szAligned)*szAligned;  // szB in element
   szC = ((M*ldc+szAligned-1)/szAligned)*szAligned;  // szC in element 

   pa = (VALUETYPE*)malloc(szA*sizeof(VALUETYPE)+ATL_Cachelen);
   assert(pa);
   a = (VALUETYPE*) ATL_AlignPtr(pa);
   
   pb = (VALUETYPE*)malloc(szB*sizeof(VALUETYPE)+ATL_Cachelen);
   assert(pb);
   b = (VALUETYPE*) ATL_AlignPtr(pb);
   
   pc = (VALUETYPE*)malloc(szC*sizeof(VALUETYPE)+ATL_Cachelen);
   assert(pc);
   c = (VALUETYPE*) ATL_AlignPtr(pc); 
#ifdef PTTIME
   #pragma omp parallel for schedule(static)
#endif
   for (i=0; i < szA; i++)
      a[i] = distribution(generator);  
#ifdef PTTIME
   #pragma omp parallel for schedule(static)
#endif
   for (i=0; i < szB; i++)
      b[i] = distribution(generator);  
#ifdef PTTIME
   #pragma omp parallel for schedule(static)
#endif
   for (i=0; i < szC; i++)
      c[i] = distribution(generator);  
/*
 *    To make rowptr, colids, values non-readonly 
 *    We may use it later if we introduce an inspector phase 
 *    NOTE: MKL uses diff type system ..
 */
      rowptr = (IT*) malloc((M+1)*sizeof(IT));
      assert(rowptr);
#ifdef PTTIME
   #pragma omp parallel for schedule(static)
#endif
      for (i=0; i < M+1; i++)
         rowptr[i] = S.rowptr[i];
   
      colids = (IT*) malloc(S.nnz*sizeof(IT));
      assert(colids);
#ifdef PTTIME
   #pragma omp parallel for schedule(static)
#endif
      for (i=0; i < S.nnz; i++)
         colids[i] = S.colids[i]; 
      
      values = (VALUETYPE*) malloc(S.nnz*sizeof(VALUETYPE));
      assert(values);
#ifdef PTTIME
   #pragma omp parallel for schedule(static)
#endif
      for (i=0; i < S.nnz; i++)
         values[i] = distribution(generator);  
/*
 *    NOTE: with small working set, we should not skip the first iteration 
 *    (warm cache), because we want to time out of cache... 
 *    We run this timer either for in-cache data or large working set
 *    So we can safely skip 1st iteration... C will be in cache then
 */

   results = CSR_TIMER(tkern, nrep, M, N, K, alpha, nnz, rows, cols, values, 
                       rowptr, colids, a, lda, b, ldb, beta, c, ldc); 
   free(rowptr);
   free(colids);
   free(values);
   free(pb);
   free(pc);
   
   return(results);
}
/*
 * Cache Flushing:  
 */
template<typename IT, csr_timer_cf_t<IT> CSR_TIMER>
vector <double> doCFTiming_Acsr
(
 const CSR<INDEXTYPE, VALUETYPE> &S, 
 IT M, 
 IT N, 
 IT K,
 const VALUETYPE alpha,
 const VALUETYPE beta,
 const int csKB,
 const int nrep,
 const int tkern
 )
{
   int szAligned; 
   IT i, j;
   vector <double> results; 
   double start, end;
   IT nnz, rows, cols;
   //size_t szB, szC, ldb, ldc; 
   IT szA, szB, szC, lda, ldb, ldc; 
   IT szM, szNNZ, csz, dsz, ndsets;
   IT nisets, isz;
   VALUETYPE *vp, *vip;
   VALUETYPE *pa, *a, *pb, *b, *pc, *c, *values;
   IT *rowptr, *colids;

#if 0
#if defined(PTTIME) && defined(NTHREADS)
   omp_set_num_threads(NTHREADS);
#endif
#endif
   //printf("---Calling cache flushing timer\n");

   std::default_random_engine generator;
   std::uniform_real_distribution<double> distribution(0.0,1.0);

   lda = ldb = ldc = K; // considering all row-major   
/*
 * NOTE: we are considering two work set here for two types: 
 *    VALUETYPE, INDEXTYPE
 * Each workset will occupy at least the size of csKB. So, if csKB is size of
 * last level cache, total workset will be double the cache size.
 * Since we can not allocate memory for two different datatype in a same 
 * workspace, we have to do it seperately.
 */
   csz = csKB*1024/sizeof(VALUETYPE);
   szAligned = ATL_Cachelen / sizeof(VALUETYPE);
   szA = ((M*ldb+szAligned-1)/szAligned)*szAligned;  // szB in element
   szB = ((N*ldb+szAligned-1)/szAligned)*szAligned;  // szB in element
   szC = ((M*ldc+szAligned-1)/szAligned)*szAligned;  // szC in element 
   szNNZ = ((S.nnz+szAligned-1)/szAligned)*szAligned;  // szC in element 
   szM = ((M+szAligned-1)/szAligned)*szAligned;  // szC in element 

   /* for VALUETYPE */
   dsz = szA + szB + szC + szNNZ + 4*ATL_Cachelen;
   ndsets = (csz + dsz -1)/dsz;
   if (ndsets < 1) ndsets = 1;
   /* for INDEXTYPE */
   isz = szM + szNNZ + 2*ATL_Cachelen;
   nisets = (csz + isz -1)/isz;
   if (nisets < 1) nisets = 1;
#if 0
   fprintf(stderr, "**** csKB=%d, csz=%d, ndsets=%ld, nisets=%ld, dsz=%d, isz=%d\n",
         csKB, csz, ndsets, nisets, dsz, isz);
   fflush(stderr);
#endif
/*
 * making the workspace of each of the array at least csKB (cache size)
 * assumption: Memory size is atleast as large as to hold all the workspace
 */
   vp = (VALUETYPE*)malloc(ndsets*dsz*sizeof(VALUETYPE));
   assert(vp);
   a = (VALUETYPE*) ATL_AlignPtr(vp);
   b = a + szA; 
   c = b + szB;
   values = c + szC;
   // OPS!!! rowptr is INT pointer!!! 
   vip = (VALUETYPE*)malloc(nisets*isz*sizeof(IT));
   assert(vip);
   rowptr = (IT*) ATL_AlignPtr(vip);
   colids = rowptr + szM;
/*
 * initialize all working set
 * NOTE: the idea here is that each part of the working will be sz apart
 *    set1_a = set0_a + sz
 *    set1_b = set0_b + sz
 *    set1_c = set0_c + sz
 */
  
   for (j=0; j < ndsets; j++)
   {
      for (i=0; i < szA; i++)
         a[i+j*dsz] = distribution(generator);  
      for (i=0; i < szB; i++)
         b[i+j*dsz] = distribution(generator);  
      for (i=0; i < szC; i++)
         c[i+j*dsz] = distribution(generator);  
      for (i=0; i < S.nnz; i++)
         values[i+j*dsz] = distribution(generator);
   }
   for (j=0; j < nisets; j++)
   {
      for (i=0; i < M+1; i++)
         rowptr[i+j*isz] = S.rowptr[i];
      for (i=0; i < S.nnz; i++)
         colids[i+j*isz] = S.colids[i]; 
   }
/*
 *    NOTE: with small working set, we should not skip the first iteration 
 *    (warm cache), because we want to time out of cache... 
 *    We run this timer either for in-cache data or large working set
 *    So we can safely skip 1st iteration... C will be in cache then
 */

   results = CSR_TIMER(ndsets, dsz, nisets, isz, nrep, tkern, M, N, K, 
         alpha, nnz, rows, cols, values, rowptr, colids, a, lda, b, ldb, 
         beta, c, ldc); 
   free(vp);
   free(vip);
   
   return(results);
}

/*
 * Run both trusted and test timer and compare results 
 */
void GetSpeedup(string inputfile, int option, INDEXTYPE M, 
      INDEXTYPE K, int csKB, int nrep, int isTest, int skipHeader, 
      VALUETYPE alpha, VALUETYPE beta, int tkern)
{
   int nerr, norandom;
   INDEXTYPE i;
   vector<double> res0, res1; 
   double exeTime0, exeTime1, inspTime0, inspTime1; 
   INDEXTYPE N, blkid; /* A->MxN, B-> NxD, C-> MxD */
   vector <INDEXTYPE> rblkids;
   CSR<INDEXTYPE, VALUETYPE> S_csr0; 
   CSR<INDEXTYPE, VALUETYPE> S_csr1; 
   CSC<INDEXTYPE, VALUETYPE> S_csc;
   

   SetInputMatricesAsCSC(S_csc, inputfile);
   S_csc.Sorted(); 
   N = S_csc.cols; 
   
   //cout << "K = " << K << endl; 
   // genetare CSR version of A  
   S_csr0.make_empty(); 
   S_csr0 = *(new CSR<INDEXTYPE, VALUETYPE>(S_csc));
   S_csr0.Sorted();
  /*
   * check for valid M.
   * NOTE: rows and cols of sparse matrix can be different 
   */
   if (!M || M > S_csr0.rows)
      M = S_csr0.rows;
/*
 * test the result if mandated 
 * NOTE: general notation: 
 *          Sparse Matrix : S -> MxN 
 *          Dense Matrix  : A->MxK B->NxK, C->MxK
 */
   assert(N && M && K);
   if (isTest)
   {
#if 0
   #ifdef DREAL
      nerr = doTesting_Acsr<mytrusted_csr, dgfusedMM_csr>
                               (S_csr0, M, N, K, alpha, beta, tkern); 
   #else
      nerr = doTesting_Acsr<mytrusted_csr, sgfusedMM_csr>
                               (S_csr0, M, N, K, alpha, beta, tkern); 
   #endif
#else // calling general fusedmm 
      nerr = doTesting_Acsr<mytrusted_csr, mytest_csr>
                               (S_csr0, M, N, K, alpha, beta, tkern); 
#endif
      // error checking 
      if (!nerr)
         fprintf(stdout, "PASSED TEST\n");
      else
      {
         fprintf(stdout, "FAILED TEST, %d ELEMENTS\n", nerr);
         exit(1); // test failed, not timed 
      }

   }
/*
 * Now, it's time to add timer 
 */
   inspTime0 = inspTime1 = exeTime0 = exeTime1 = 0.0;
   {
#ifdef TIME_MKL
      // call Trusted mkl code 
      assert(tkern == 'm'); // only spmm 
      res0 = doTiming_Acsr<MKL_INT, callTimerMKL_Acsr>(S_csr0, M, N, K, 
                  alpha, beta, csKB, nrep, tkern);
#else
      // call Trusted ... c code 
      //res0 = doTiming_Acsr<INDEXTYPE, callTimerTrusted_Acsr>(S_csr0, M, N, K, 
      //            alpha, beta, csKB, nrep, tkern);
      res0 = doCFTiming_Acsr<INDEXTYPE, callCFTimerTrusted_Acsr>(S_csr0, M, N, K, 
                  alpha, beta, csKB, nrep, tkern);
#endif
      inspTime0 += res0[0];
      exeTime0 += res0[1];
      
      //res1 = doTiming_Acsr<INDEXTYPE, callTimerTest_Acsr>(S_csr0, M, N, K, 
      //            alpha, beta, csKB, nrep, tkern);
      res1 = doCFTiming_Acsr<INDEXTYPE, callCFTimerTest_Acsr>(S_csr0, M, N, K, 
                  alpha, beta, csKB, nrep, tkern);
      //cout << "      blkid = " << blkid << " ExeTime = " << res1[1] << endl;    
      inspTime1 += res1[0];
      exeTime1 += res1[1];
   }
   //inspTime0 /= nrblk; 
   //inspTime1 /= nrblk; 
   
    //exeTime0 /= nrblk; 
   //exeTime1 /= nrblk; 
   
   if(!skipHeader) 
   {
      cout << "Filename,"
         << "NNZ,"
         << "M,"
         << "N,"
         << "K,"
#ifdef TIME_MKL
         << "Trusted_inspect_time,"
#endif
         << "Trusted_exe_time,"
#ifdef TIME_MKL
         << "Test_inspect_time,"
#endif
         << "Test_exe_time,"
         << "Speedup_exe_time,"
#ifdef TIME_MKL 
         << "Speedup_total,"
         << "Critical_point" 
#endif
         << endl;
   }
#ifdef TIME_MKL 
   double critical_point = (res0[0]/(res1[1]-res0[1])) < 0.0 ?  -1.0 
                                             : (res0[0]/(res1[1]-res0[1])); 
#endif
   cout << inputfile << "," 
        << S_csr0.nnz << "," 
        << M << "," 
        << N << "," 
        << K << "," << std::scientific
#ifdef TIME_MKL 
        << inspTime0 << "," 
#endif
        << exeTime0 << "," 
#ifdef TIME_MKL 
        << inspTime1 << "," 
#endif
        << exeTime1 << "," 
        << std::fixed << std::showpoint
        << exeTime0/exeTime1
#ifdef TIME_MKL
        << ","
        << ((inspTime0+exeTime0)/(inspTime1+exeTime1)) << ","  
        << critical_point
#endif
        << endl;
}

void Usage()
{
   printf("\n");
   printf("Usage for CompAlgo:\n");
   printf("-input <string>, full path of input file (required).\n");
   printf("-M <number>, rows of S (can be less than actual rows of S).\n");
   printf("-K <number>, number of cols of A, B and C \n");
   printf("-C <number>, Cachesize in KB to flush it for small workset \n");
   printf("-nrep <number>, number of repeatation \n");
   printf("-nrblk <number>, number of random blk with row M, 0/-1: all  \n");
   printf("-T <0,1>, 1 means, run tester as well  \n");
   printf("-t <t,s>, t : t-distribution, s : sigmoid  \n");
   printf("-skHd<1>, 1 means, skip header of the printed results  \n");
   printf("-trusted <option#>\n" 
          "   1)MKL 2)FUSEDMM_UNOPTIMIZED\n");
   //printf("-test <option#>\n"
   //       "   1)MKL 2)CSR_IKJ 3)CSR_KIJ 4)CSR_IKJ_D128 5)CSR_KIJ_D128\n");
   printf("-ialpha <1, 0, 2>, alpha respectively 1.0, 0.0, X  \n");
   printf("-ibeta <1, 0, 2>, beta respectively 1.0, 0.0, X \n");
   printf("-h, show this usage message  \n");

}
void GetFlags(int narg, char **argv, string &inputfile, int &option, 
      INDEXTYPE &M, INDEXTYPE &K, int &csKB, int &nrep, 
      int &isTest, int &skHd, VALUETYPE &alpha, VALUETYPE &beta, char &tkern)
{
   int ialpha, ibeta; 
/*
 * default values 
 */
   option = 1; 
   inputfile = "";
   K = 128; 
   M = 0;
/*
 * default kernel based on macro now
 */
#if defined(SIGMOID_UDEF)
   tkern = 's';
#elif defined(TDIST_UDEF)
   tkern = 't';
#elif defined(GCN_UDEF)
   tkern = 'g';
#elif defined(SPMM_UDEF)
   tkern = 'm';
#elif defined(FR_UDEF)
   tkern = 'f';
#else
   tkern = 's';
#endif

   isTest = 0; 
   nrep = 20;
   //nrblk = 1;
   skHd = 0; // by default print header
   csKB = 25344; // L3 in KB 
   
   // alphaX, betaX would be the worst case for our implementation  
   ialpha=1; 
   alpha=1.0; 
   ibeta=1; 
   //beta = 1.0;
   
   for(int p = 1; p < narg; p++)
   {
      if(strcmp(argv[p], "-input") == 0)
      {
	 inputfile = argv[p+1];
      }
      else if(strcmp(argv[p], "-option") == 0)
      {
	 option = atoi(argv[p+1]);
      }
      else if(strcmp(argv[p], "-K") == 0)
      {
	 K = atoi(argv[p+1]);
      }
      else if(strcmp(argv[p], "-M") == 0)
      {
	 M = atoi(argv[p+1]);
      }
      else if(strcmp(argv[p], "-C") == 0)
      {
	 csKB = atoi(argv[p+1]);
      }
      else if(strcmp(argv[p], "-nrep") == 0)
      {
	 nrep = atoi(argv[p+1]);
      }
      else if(strcmp(argv[p], "-T") == 0)
      {
	 isTest = atoi(argv[p+1]);
      }
      else if(strcmp(argv[p], "-t") == 0)
      {
	 tkern = argv[p+1][0];
      }
      else if(strcmp(argv[p], "-skHd") == 0)
      {
	 skHd = atoi(argv[p+1]);
      }
      else if(strcmp(argv[p], "-ialpha") == 0)
      {
	 ialpha = atoi(argv[p+1]);
      }
      else if(strcmp(argv[p], "-ibeta") == 0)
      {
	 ibeta = atoi(argv[p+1]);
      }
      else if(strcmp(argv[p], "-h") == 0)
      {
         Usage();
         exit(1);
      }
   }
   if (inputfile == "")
   {
      cout << "Need input file ??? " << endl;
      exit(1);
   }
/*
 * set alpha beta
 */
#if 0 
   if (ialpha == 1 && ibeta == 1)
   {
      alpha = 1.0; 
      beta = 1.0;
   }
   else if (ialpha == 2 && ibeta == 2 )
   {
      alpha = 2.0; 
      beta = 2.0;
   }
   else
   {
      cout << "ialpha =  " << ialpha << " ibeta = " << ibeta << " not supported"
         << endl;
      exit(1);
   }
#endif
/*
 * supported beta = 0 and beta = 1 case
 */
   if (ibeta = 0)
      beta = 0.0;
   else
      beta = 1.0;

}
int main(int narg, char **argv)
{
   INDEXTYPE M, K;
   VALUETYPE alpha, beta;
   int option, csKB, nrep, isTest, skHd, nrblk;
   char tkern;
   string inputfile; 
   GetFlags(narg, argv, inputfile, option, M, K, csKB, nrep, isTest, skHd, 
            alpha, beta, tkern);
   GetSpeedup(inputfile, option, M, K, csKB, nrep, isTest, skHd, alpha, beta, 
         tkern);
   return 0;
}
