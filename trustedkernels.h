#ifndef TRUSTED_KERNEL_H
#define TRUSTED_KERNEL_H 
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
/*
 * implementation of general but unoptimized kernel 
 */
/*
 * VOP message 
 */
/* 1st four bit is reserved for VOP operation, !0 = binary operation */
#define VOP_COPY 0x0  /* output just the lhs vector */
#define VOP_ADD  0x1
#define VOP_SUB  0x2
#define VOP_MAX  0x3
#define VOP_MIN  0x4
#define VOP_UDEF  0xF
#define VOP_CLEAR (bvec) (((bvec)>>4)<<4)  /* clearing last 4 bit */
#define VOP_MASK (bvec) (bvec & 0xF)
// set op destructive, overwritten imsg
#define SET_VOP_FLAG (imsg, vflag)  (imsg = (VOP_CLEAR(imsg) | vflag)) 
// get readonly  
#define GET_VOP_FLAG (imsg) VOP_MASK(imsg)
/*
 * ROP message
 * 2nd four bit is reserved for ROP 
 */
#define ROP_NOOP 0x00  /* not applied ... can skip the function */
#define ROP_DOT 0x10   /* dot product */
#define ROP_ADD 0x20   /* sum, only consider the lhs */
#define ROP_UDEF 0xF0   /* sum, only consider the lhs */

#define ROP_CLEAR (bvec) ((bvec) & (~((int32_t)0xF0)))  /* clearing end 4 bit */
#define ROP_MASK (bvec) ((bvec) & 0xF0)  /* masking out 2nd 4 bit */

#define SET_ROP_FLAG (imsg, vflag)  (imsg = (ROP_CLEAR(imsg) | vflag)) 
#define GET_ROP_FLAG (imsg) ROP_MASK(imsg)
/*
 * SOP message : 3rd 4bit 
 */
#define SOP_NOOP 0x000  /* not applied ... can skip the function */
#define SOP_UDEF 0xF00  /* not applied ... can skip the function */

#define SOP_CLEAR (bvec) ((bvec) & (~((int32_t)0xF00)))  /* clearing end 4 bit */
#define SOP_MASK (bvec) ((bvec) & 0xF00)  /* masking out 2nd 4 bit */

#define SET_SOP_FLAG (imsg, vflag)  (imsg = (SOP_CLEAR(imsg) | vflag)) 
#define GET_SOP_FLAG (imsg) SOP_MASK(imsg)
/*
 * VSC message : 4th 4-bit 
 */
#define VSC_NOOP 0x0000  /* not applied ... can skip the function */
#define VSC_MUL 0x1000   /* mul */
#define VSC_ADD 0x2000   /* sum, only consider the lhs */
#define VSC_UDEF 0xF000  /* not applied ... can skip the function */

#define VSC_CLEAR (bvec) ((bvec) & (~((int32_t)0xF000)))  /* clearing end 4 bit */
#define VSC_MASK (bvec) ((bvec) & 0xF000)  /* masking out 2nd 4 bit */

#define SET_VSC_FLAG (imsg, vflag)  (imsg = (VSC_CLEAR(imsg) | vflag)) 
#define GET_VSC_FLAG (imsg) VSC_MASK(imsg)
/*
 * VSC message : 5th 4-bit 
 */
#define AOP_NOOP 0x00000  /* not applied ... can skip the function */
#define AOP_ADD 0x10000   /* dot product */
#define AOP_MAX 0x20000   /* sum, only consider the lhs */
#define AOP_MIN 0x20000   /* sum, only consider the lhs */
#define AOP_UDEF 0xF0000  /* not applied ... can skip the function */

#define AOP_CLEAR (bvec) ((bvec) & (~((int32_t)0xF0000)))  /* clearing end 4 bit */
#define AOP_MASK (bvec) ((bvec) & 0xF0000)  /* masking out 2nd 4 bit */

#define SET_AOP_FLAG (imsg, vflag)  (imsg = (AOP_CLEAR(imsg) | vflag)) 
#define GET_AOP_FLAG (imsg) AOP_MASK(imsg)


/* function pointer type for userdef VOP operation */
typedef int (*FP_VOP_UDEF)(INDEXTYPE M, VALUETYPE *lhs, INDEXTYPE N, VALUETYPE *rhs,
      INDEXTYPE K, VALUETYPE *T); 

typedef VALUETYPE (*FP_SOP_UDEF)(VALUETYPE val); 


/* test user defined funct: just add */
int myVOPfunc(INTDEXTYPE M, VALUETYPE *lhs, INDEXTYPE N, VALUETYPE *rhs, 
      INDEXTYPE K, VALUETYPE *T)
{
   assert(M == N && T>= M);
   for (INDEXTYPE i=0; i < M; i++)
      T[i] = lhs[i] + rhs[i];
}
/* test user defined funct: just add */
VALUETYPE sigmoid(VALUETYPE val)
{
   VALUETYPE x;
   /*
   for(IdType i = 0; i < SM_TABLE_SIZE; i++){
                x = 2.0 * SM_BOUND * i / SM_TABLE_SIZE - SM_BOUND;
                sm_table[i] = 1.0 / (1 + exp(-x));
   */ 
   return 1.0;
}

/*=============================================================================
 *    vector to vector operation:
 *       input: 
 *          1. message: copy, binary op (+, -, *, max, min, user define) 
 *             NOTE: for copy we can bypass this function call 
 *          2. 1-D lhs vector : M, lhs 
 *          3. 1-D rhs vector: N, rhs
 *          4. userdefined function ptr 
 *       output: 
 *          1. output vector : K, T 
 *       function output = status ... 0 means operation successful 
 */
inline int vector-to-vector(int32_t imsg, INDEXTYPE M, VALUETYPE *lhs,  
      INDEXTYPE N, VALUETYPE *rhs, FP_VOP_UDEF opfunc, INDEXTYPE K, VALUETYPE *T)
{
   int32_t bvop; 
   bvop = GET_VOP_FLAG(imsg); /* mask out only the VOP bits */
/*
 * case: just copy
 */
   if (bvop == VOP_COPY)  /* just copy the lhs to output */
   {
      if (M != K) 
      {
#ifdef DEBUG
         fprintf(stderr, "dimension of lhs and T are not same!");
#endif
         return 2; 
      }
      for (INDEXTYPE i=0; i < M; i++)
         T[i] = lhs[i]
   }
/*
 * case : user defined funciton
 */
   if (bvop == VOP_UDEF)  /* apply user define function */
      opfunc(M, lhs, N, rhs, K, T);
/*
 * FIXME: handle all other like DGL lib 
 */

   else if (bvop == VOP_ADD)
   {
      //assert(M == N);
      if (M != N)
      {
#ifdef DEBUG
         fprintf(stderr, "dimension of lhs and rhs are not same!");
#endif
         return 2;
      }
      for (INDEXTYPE i=0; i < M; i++)
         T[i] = lhs[i] + rhs[i];
   }

   else if (bvop == VOP_SUB)
   {
      if (M != N)
      {
#ifdef DEBUG
         fprintf(stderr, "dimension of lhs and rhs are not same!");
#endif
         return 2;
      }
      for (INDEXTYPE i=0; i < M; i++)
         T[i] = lhs[i] - rhs[i];
   }
   else if (bvop == VOP_MAX)
   {
      if (M != N)
      {
#ifdef DEBUG
         fprintf(stderr, "dimension of lhs and rhs are not same!");
#endif
         return 2;
      }
      for (INDEXTYPE i=0; i < M; i++)
         T[i] = max(lhs[i], rhs[i]);
   }
   else if (bvop == VOP_MIN)
   {
      if (M != N)
      {
#ifdef DEBUG
         fprintf(stderr, "dimension of lhs and rhs are not same!");
#endif
         return 2;
      }
      for (INDEXTYPE i=0; i < M; i++)
         T[i] = min(lhs[i], rhs[i]);
   }

   return 0;
}

/*=============================================================================
 *    vector to scalar operation: ROP
 *       input: 
 *          1. message: noop : may be no need to call this function 
 *              binary op -> dot product, unary: sum, user define 
 *          2. 1-D lhs vector : M, lhs 
 *          3. 1-D rhs vector: N, rhs
 *          4. userdefined function ptr 
 *       output: 
 *          1. scalar value  
 *       function output = status ... 0 means operation successful 
 */

inline int vector-to-scalar(int32_t imsg, INDEXTYPE M, VALUETYPE *lhs,  
      INDEXTYPE N, VALUETYPE *rhs, ROP_UDEF opfunc, VALUETYPE &scalar)
{
   int32_t bvop;
   bvop = GET_VOP_FLAG(imsg); /* mask out only the VOP bits */
   
   if (bvop == ROP_DOT)
   {
      if (M != N)
      {
#ifdef DEBUG
         fprintf(stderr, "dimension of lhs and rhs are not same!");
#endif
         return 2;
      }
      scalar = 0;
      for (INDEXTYPE i=0; i < M; i++)
         scalar + = lhs[i] * rhs[i];
   }
   else if (bvop == ROP_ADD)
   {
      if (M != N)
      {
#ifdef DEBUG
         fprintf(stderr, "dimension of lhs and rhs are not same!");
#endif
         return 2;
      }
      scalar = 0;
      for (INDEXTYPE i=0; i < M; i++)
         scalar + = lhs[i];
   }
   return 0;  /* VOP done successfully */  
}

/*=============================================================================
 *    scalar function: SOP
 *    user defined
 */
inline int SOP(int32_t imsg, VALUETYPE val, FP_SOP_UDEF sopfunc, VALUETYPE &out )
{
   out = sopfunc(val);
   return 0;
}

/*=============================================================================
 *    vector scaling : VSC
 *    user defined
 */
inline int VSC(int32_t imsg, INDEXTYPE M, VALUETYPE *lhs, VALUETYPE *rhs, 
      VALUETYPE scal)
{
   if (GET_VSC_FLAG(imsg) == VSC_MUL)
   {
      for(INDEXTYPE i=0; i < M; i++)
         lhs[i] = rhs[i] * scal;
   }
   return 0;
}

/*=============================================================================
 *    vector aggrigration: AOP
 *    user defined
 */

inline int AOP(int32_t imsg, INDEXTYPE M, VALUETYPE *lhs,
      FP_VOP_UDEF opfunc, VALUETYPE *rhs)
{
   int32_t bvop;
   bvop = GET_VOP_FLAG(imsg); /* mask out only the VOP bits */

   if (bvop == AOP_ADD)
   {
      for (INDEXTYPE i=0; i < M; i++)
         lhs[i] += rhs[i];
   }
   else if (bvop == AOP_MAX)
   {
      for (INDEXTYPE i=0; i < M; i++)
         lhs[i] = max(lhs[i], rhs[i]);
   }
   else if (bvop == AOP_MAX)
   {
      for (INDEXTYPE i=0; i < M; i++)
         lhs[i] = min(lhs[i], rhs[i]);
   }

   return 0;
}

/*=============================================================================
 *       
 *       Trusted general purpose kernels
 *
 *          NOTE: for user defined function, we will use a fixed name for each 
 *          user function. like: myVOPfunc, myROPfunc, etc. may want to pass
 *          them as parameter later 
 *
 *============================================================================*/

void gtrusted_csr 
(
/* sigmoid: VOP_COPY | ROP_DOT | SOP_UDEF | VSC_MUL | AOP_ADD */ 
   const int32_t imessage,  // i
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
   const VALUETYPE *a,     // Dense X matrix
   const INDEXTYPE lda,   // 1eading dimension of b (col size since row-major)  
   const VALUETYPE *b,     // Dense Y matrix
   const INDEXTYPE ldb,   // leading`1 dimension of b (col size since row-major)  
   const VALUETYPE beta,  // beta value 
   VALUETYPE *c,           // Dense matrix Z
   const INDEXTYPE ldc    // leading dimension size of c (col size since roa-major) 
)
{
   
   #pragma omp parallel for 
   for (INDEXTYPE i = 0; i < m; i++)
   {
      VALUETYPE Ai = a + i * lda; 
      VALUETYPE Ci = a + i * ldc; 
      for (INDEXTYPE j=pntrb[i]; j < pntre[i]; j++)
      {
/*
 *       call five statge
 *          1. VOP
 *          2. ROP
 *          3. SOP
 *          4. VSC
 *          5. AOP 
 *
 */
/*
 *       a -> m x k, b -> n x k, c -> m x k, S -> m x n  
 */
         VALUETYPE T[k];
         INDEXTYPE cid = indx[j];
         VALUETYPE Bj = b + cid * ldb; 
         VALUETYPE *lhs = Ai;
         VALUETYPE *rhs = Bj;
         vector-to-vector(VOP_COPY, k, lhs, 0, NULL, NULL, k, T);

         VALUETYPE scal, out; 
         vector-to-scalar(ROP_DOT, k, T, k, Bj, NULL, scal);
         SOP(SOP_UDEF, scal, sigmoid, out);

         VSC(VSC_MUL, k, T, Bj, out);

         AOP(AOP_ADD, k, Ci, NULL, T ); 

      }
   }
}

#endif
