#include <iostream>
#include <algorithm>
#include <omp.h>

#ifdef DREAL
   #define VALUETYPE double
#else
   #define VALUETYPE float
#endif
#include "fusedMM.h"
#include "fusedMM_internal.h"


#define fmax(x,y) ( (x) > (y) ? (x) : (y))
#define fmin(x,y) ( (x) < (y) ? (x) : (y))
#define fabs(x) ( (x) >= 0 ? (x) : -(y))


/*============================================================================
 *    VOP operations 
 */

int KERN_VOP_COPY_LHS (INDEXTYPE lhs_dim, const VALUETYPE *lhs, INDEXTYPE rhs_dim, 
      const VALUETYPE *rhs, INDEXTYPE out_dim, VALUETYPE *out)
{
#ifdef DEBUG 
   if (lhs_dim != out_dim)
   {
      fprintf(stderr, "dimension of lhs and rhs are not same!!!");
      return FUSEDMM_FAIL_RETURN; 
   }
#endif
   for (INDEXTYPE i = 0; i < out_dim; i++)
      out[i] = lhs[i];

   return FUSEDMM_SUCCESS_RETURN;
}

int KERN_VOP_COPY_RHS (INDEXTYPE lhs_dim, const VALUETYPE *lhs, INDEXTYPE rhs_dim, 
      const VALUETYPE *rhs, INDEXTYPE out_dim, VALUETYPE *out)
{
#ifdef DEBUG 
   if (lhs_dim != out_dim)
   {
      fprintf(stderr, "dimension of lhs and rhs are not same!!!");
      return FUSEDMM_FAIL_RETURN; 
   }
#endif
   for (INDEXTYPE i = 0; i < out_dim; i++)
      out[i] = rhs[i];

   return FUSEDMM_SUCCESS_RETURN;
}

int KERN_VOP_ADD (INDEXTYPE lhs_dim, const VALUETYPE *lhs, INDEXTYPE rhs_dim, 
      const VALUETYPE *rhs, INDEXTYPE out_dim, VALUETYPE *out)
{
#ifdef DEBUG 
   if (lhs_dim != rhs_dim && lhs_dim != out_dim)
   {
      fprintf(stderr, "dimension of lhs and rhs are not same!!!");
      return FUSEDMM_FAIL_RETURN; 
   }
#endif
   for (INDEXTYPE i = 0; i < out_dim; i++)
      out[i] = lhs[i] + rhs[i];

   return FUSEDMM_SUCCESS_RETURN;
}

int KERN_VOP_SUB (INDEXTYPE lhs_dim, const VALUETYPE *lhs, INDEXTYPE rhs_dim, 
      const VALUETYPE *rhs, INDEXTYPE out_dim, VALUETYPE *out)
{
#ifdef DEBUG 
   if (lhs_dim != rhs_dim && lhs_dim != out_dim)
   {
      fprintf(stderr, "dimension of lhs and rhs are not same!!!");
      return FUSEDMM_FAIL_RETURN; 
   }
#endif
   for (INDEXTYPE i = 0; i < out_dim; i++)
      out[i] = - lhs[i] + rhs[i]; //changed: previously it was: out[i] = lhs[i] + rhs[i]

   return FUSEDMM_SUCCESS_RETURN;
}

int KERN_VOP_MAX (INDEXTYPE lhs_dim, const VALUETYPE *lhs, INDEXTYPE rhs_dim, 
      const VALUETYPE *rhs, INDEXTYPE out_dim, VALUETYPE *out)
{
#ifdef DEBUG 
   if (lhs_dim != rhs_dim && lhs_dim != out_dim)
   {
      fprintf(stderr, "dimension of lhs and rhs are not same!!!");
      return FUSEDMM_FAIL_RETURN; 
   }
#endif
   for (INDEXTYPE i = 0; i < out_dim; i++)
      out[i] = fmax(lhs[i], rhs[i]);

   return FUSEDMM_SUCCESS_RETURN;
}

int KERN_VOP_MIN (INDEXTYPE lhs_dim, const VALUETYPE *lhs, INDEXTYPE rhs_dim, 
      const VALUETYPE *rhs, INDEXTYPE out_dim, VALUETYPE *out)
{
#ifdef DEBUG 
   if (lhs_dim != rhs_dim && lhs_dim != out_dim)
   {
      fprintf(stderr, "dimension of lhs and rhs are not same!!!");
      return FUSEDMM_FAIL_RETURN; 
   }
#endif
   for (INDEXTYPE i = 0; i < out_dim; i++)
      out[i] = fmin(lhs[i], rhs[i]);

   return FUSEDMM_SUCCESS_RETURN;
}

/* ============================================================================
 *    ROP operation  
 * 
 */
int KERN_ROP_NOOP (INDEXTYPE lhs_dim, const VALUETYPE *lhs, INDEXTYPE rhs_dim, 
      const VALUETYPE *rhs, VALUETYPE &out)
{
   /* Dummy function : no operation */
   return FUSEDMM_SUCCESS_RETURN;
}

int KERN_ROP_DOT (INDEXTYPE lhs_dim, const VALUETYPE *lhs, INDEXTYPE rhs_dim, 
      const VALUETYPE *rhs, VALUETYPE &out)
{
#ifdef DEBUG 
   if (lhs_dim != rhs_dim)
   {
      fprintf(stderr, "dimension of lhs and rhs are not same!!!");
      return FUSEDMM_FAIL_RETURN; 
   }
#endif
   out = 0.0;
   for (INDEXTYPE i = 0; i < lhs_dim; i++)
      out += lhs[i]*rhs[i];

   return FUSEDMM_SUCCESS_RETURN;
}

int KERN_ROP_ADD_LHS (INDEXTYPE lhs_dim, const VALUETYPE *lhs, INDEXTYPE rhs_dim, 
      const VALUETYPE *rhs, VALUETYPE &out)
{
   out = 0.0;
   for (INDEXTYPE i = 0; i < lhs_dim; i++)
      out += lhs[i];
   return FUSEDMM_SUCCESS_RETURN;
}

int KERN_ROP_ADD_RHS (INDEXTYPE lhs_dim, const VALUETYPE *lhs, INDEXTYPE rhs_dim, 
      const VALUETYPE *rhs, VALUETYPE &out)
{
   out = 0.0;
   for (INDEXTYPE i = 0; i < rhs_dim; i++)
      out += rhs[i];
   return FUSEDMM_SUCCESS_RETURN;
}

/* ============================================================================
 *    SOP operation  
 * 
 */
int KERN_SOP_NOOP(VALUETYPE val, VALUETYPE &out)
{
   // dummy function to avoid extra checking 
   return FUSEDMM_SUCCESS_RETURN;
}
/* ============================================================================
 *    VSC operation  
 * 
 */

int KERN_VSC_NOOP(INDEXTYPE rhs_dim, const VALUETYPE *rhs, VALUETYPE scal, 
      INDEXTYPE out_dim, VALUETYPE *out) 
{
   // dummy function to avoid extra checking 
   return FUSEDMM_SUCCESS_RETURN;
}

int KERN_VSC_MUL(INDEXTYPE rhs_dim, const VALUETYPE *rhs, VALUETYPE scal, 
      INDEXTYPE out_dim, VALUETYPE *out) 
{
#ifdef DEBUG 
   if (rhs_dim != out_dim)
   {
      fprintf(stderr, "dimension of lhs and rhs are not same!!!");
      return FUSEDMM_FAIL_RETURN; 
   }
#endif
   for (INDEXTYPE i = 0; i < rhs_dim; i++)
      out[i] = scal * rhs[i];
   
   return FUSEDMM_SUCCESS_RETURN;
}

int KERN_VSC_ADD(INDEXTYPE rhs_dim, const VALUETYPE *rhs, VALUETYPE scal, 
      INDEXTYPE out_dim, VALUETYPE *out) 
{
#ifdef DEBUG 
   if (rhs_dim != out_dim)
   {
      fprintf(stderr, "dimension of lhs and rhs are not same!!!");
      return FUSEDMM_FAIL_RETURN; 
   }
#endif
   for (INDEXTYPE i = 0; i < rhs_dim; i++)
      out[i] = scal + rhs[i];
   
   return FUSEDMM_SUCCESS_RETURN;
}

/* ============================================================================
 *    AOP operation  
 * 
 */

//int KERN_AOP_NOOP(INDEXTYPE lhs_dim, const VALUETYPE *lhs, INDEXTYPE rhs_dim, 
//      const VALUETYPE *rhs, INDEXTYPE out_dim, VALUETYPE *out) 
int KERN_AOP_NOOP(INDEXTYPE rhs_dim, const VALUETYPE *rhs, INDEXTYPE out_dim, 
      VALUETYPE *out) 
{
   // dummy function to avoid extra checking 
   return FUSEDMM_SUCCESS_RETURN;
}

//int KERN_AOP_MUL(INDEXTYPE lhs_dim, const VALUETYPE *lhs, INDEXTYPE rhs_dim, 
//      const VALUETYPE *rhs, INDEXTYPE out_dim, VALUETYPE *out) 
int KERN_AOP_MUL(INDEXTYPE rhs_dim, const VALUETYPE *rhs, INDEXTYPE out_dim, 
      VALUETYPE *out) 
{
#ifdef DEBUG 
   if (rhs_dim != out_dim)
   {
      fprintf(stderr, "dimension of lhs and rhs are not same!!!");
      return FUSEDMM_FAIL_RETURN; 
   }
#endif
   for (INDEXTYPE i = 0; i < rhs_dim; i++)
      out[i] *= rhs[i];
   return FUSEDMM_SUCCESS_RETURN;
}

//int KERN_AOP_ADD(INDEXTYPE lhs_dim, const VALUETYPE *lhs, INDEXTYPE rhs_dim, 
//      const VALUETYPE *rhs, INDEXTYPE out_dim, VALUETYPE *out) 
int KERN_AOP_ADD(INDEXTYPE rhs_dim, const VALUETYPE *rhs, INDEXTYPE out_dim, 
      VALUETYPE *out) 
{
#ifdef DEBUG 
   if (rhs_dim != out_dim)
   {
      fprintf(stderr, "dimension of lhs and rhs are not same!!!");
      return FUSEDMM_FAIL_RETURN; 
   }
#endif
   for (INDEXTYPE i = 0; i < rhs_dim; i++)
      out[i] += rhs[i];
   return FUSEDMM_SUCCESS_RETURN;
}

//int KERN_AOP_MAX(INDEXTYPE lhs_dim, const VALUETYPE *lhs, INDEXTYPE rhs_dim, 
//      const VALUETYPE *rhs, INDEXTYPE out_dim, VALUETYPE *out) 
int KERN_AOP_MAX(INDEXTYPE rhs_dim, const VALUETYPE *rhs, INDEXTYPE out_dim, 
      VALUETYPE *out) 
{
#ifdef DEBUG 
   if (rhs_dim != out_dim)
   {
      fprintf(stderr, "dimension of lhs and rhs are not same!!!");
      return FUSEDMM_FAIL_RETURN; 
   }
#endif
   for (INDEXTYPE i = 0; i < rhs_dim; i++)
      out[i] = fmax (out[i], rhs[i]);
   return FUSEDMM_SUCCESS_RETURN;
}

//int KERN_AOP_MIN(INDEXTYPE lhs_dim, const VALUETYPE *lhs, INDEXTYPE rhs_dim, 
//      const VALUETYPE *rhs, INDEXTYPE out_dim, VALUETYPE *out) 
int KERN_AOP_MIN(INDEXTYPE rhs_dim, const VALUETYPE *rhs, INDEXTYPE out_dim, 
      VALUETYPE *out) 
{
#ifdef DEBUG 
   if (rhs_dim != out_dim)
   {
      fprintf(stderr, "dimension of lhs and rhs are not same!!!");
      return FUSEDMM_FAIL_RETURN; 
   }
#endif
   for (INDEXTYPE i = 0; i < rhs_dim; i++)
      out[i] = fmin (out[i], rhs[i]);
   return FUSEDMM_SUCCESS_RETURN;
}


/*=============================================================================
 * Select funciton based on imeesage 
 *
 */

FP_VOP_FUNC GetVOPFunc(int32_t msg)
{
   FP_VOP_FUNC VOP_FUNC; 
   switch(msg)
   {
      /* no way, need to update T  
       * case VOP_NOOP: 
         VOP_FUNC = NULL;
         break;
      */
      case VOP_COPY_LHS: 
         VOP_FUNC = KERN_VOP_COPY_LHS;
         break;
      case VOP_COPY_RHS: 
         VOP_FUNC = KERN_VOP_COPY_RHS;
         break;
      case VOP_ADD: 
         VOP_FUNC = KERN_VOP_ADD;
         break;
      case VOP_SUB: 
         VOP_FUNC = KERN_VOP_SUB;
         break;
      case VOP_MAX: 
         VOP_FUNC = KERN_VOP_MAX;
         break;
      case VOP_MIN: 
         VOP_FUNC = KERN_VOP_MIN;
         break;
      case VOP_UDEF: 
         VOP_FUNC = VOP_UDEF_FUNC;
         break;
      defautl:
         std::cout << "Unknown VOP Message" << std::endl;
         return NULL; 
   }
   return VOP_FUNC;
}

FP_ROP_FUNC GetROPFunc(int32_t msg)
{
   FP_ROP_FUNC ROP_FUNC;
   switch(msg)
   {
      case ROP_NOOP: 
         ROP_FUNC = KERN_ROP_NOOP;
         break;
      case ROP_DOT: 
         ROP_FUNC = KERN_ROP_DOT;
         break;
      case ROP_ADD_LHS: 
         ROP_FUNC = KERN_ROP_ADD_LHS;
         break;
      case ROP_ADD_RHS: 
         ROP_FUNC = KERN_ROP_ADD_RHS;
         break;
      case ROP_UDEF: 
         ROP_FUNC = ROP_UDEF_FUNC;
         break;
      defautl:
         std::cout << "Unknown ROP Message" << std::endl;
         return NULL; 
   }
   return(ROP_FUNC);
}

FP_SOP_FUNC GetSOPFunc(int32_t msg)
{
   FP_SOP_FUNC SOP_FUNC;
   switch(msg)
   {
      case SOP_NOOP: 
         SOP_FUNC = KERN_SOP_NOOP;
         break;
      case SOP_UDEF: 
         SOP_FUNC = SOP_UDEF_FUNC;
         break;
      defautl:
         std::cout << "Unknown SOP Message" << std::endl;
         return NULL; 
   }
   return(SOP_FUNC);
}

FP_VSC_FUNC GetVSCFunc(int32_t msg)
{
   FP_VSC_FUNC VSC_FUNC;
   switch(msg)
   {
      case VSC_NOOP: 
         VSC_FUNC = KERN_VSC_NOOP;
         break;
      case VSC_MUL: 
         VSC_FUNC = KERN_VSC_MUL;
         break;
      case VSC_ADD: 
         VSC_FUNC = KERN_VSC_ADD;
         break;
      case VSC_UDEF: 
         VSC_FUNC = VSC_UDEF_FUNC;
         break;
      defautl:
         std::cout << "Unknown VSC Message" << std::endl;
         return NULL; 
   }
   return(VSC_FUNC);
}

FP_AOP_FUNC GetAOPFunc(int32_t msg)
{
   FP_AOP_FUNC AOP_FUNC;
   switch(msg)
   {
      case AOP_NOOP: 
         AOP_FUNC = KERN_AOP_NOOP;
         break;
      case AOP_ADD: 
         AOP_FUNC = KERN_AOP_ADD;
         break;
      case AOP_MAX: 
         AOP_FUNC = KERN_AOP_MAX;
         break;
      case AOP_MIN: 
         AOP_FUNC = KERN_AOP_MIN;
         break;
      case AOP_UDEF: 
         AOP_FUNC = AOP_UDEF_FUNC;
         break;
      defautl:
         std::cout << "Unknown AOP Message" << std::endl;
         return NULL; 
   }
   return(AOP_FUNC);
}

int fusedMM_csr 
(
   const int32_t imessage, // message to dictate the operations  
   const INDEXTYPE m,         // number of row of X
   const INDEXTYPE n,         // number of row of Y
   const INDEXTYPE k,         // dimension (col of X or Y)
   const VALUETYPE alpha,   // not used yet
   const INDEXTYPE nnz,       // nonzeros in sparse matrix 
   const INDEXTYPE rows,      // number of rows in sparse matrix
   const INDEXTYPE cols,      // number of columns in sparse matrix 
   const VALUETYPE *val,    // value of non-zeros 
   const INDEXTYPE *indx,     // colids -> column indices 
   const INDEXTYPE *pntrb,    // starting of rowptr for each row
   const INDEXTYPE *pntre,    // ending of rowptr for each row
   const VALUETYPE *x,      // Dense X matrix
   const INDEXTYPE ldx,       // 1eading dimension of a   
   const VALUETYPE *y,      // Dense Y matrix
   const INDEXTYPE ldy,       // leading dimension of b   
   const VALUETYPE beta,    // beta value 
   VALUETYPE *z,            // Dense matrix Z
   const INDEXTYPE ldz        // leading dimension size of c  
)
{
   int status = 0;
/*
 * Select appropriate operation based on the message
 */
   FP_VOP_FUNC VOP_FUNC = GetVOPFunc(GET_VOP_FLAG(imessage));
   if(!VOP_FUNC)
      return FUSEDMM_VOP_FAIL_RETURN;

   FP_ROP_FUNC ROP_FUNC = GetROPFunc(GET_ROP_FLAG(imessage));
   if(!ROP_FUNC)
      return FUSEDMM_ROP_FAIL_RETURN;

   FP_SOP_FUNC SOP_FUNC = GetSOPFunc(GET_SOP_FLAG(imessage));
   if(!SOP_FUNC)
      return FUSEDMM_SOP_FAIL_RETURN;

   FP_VSC_FUNC VSC_FUNC = GetVSCFunc(GET_VSC_FLAG(imessage));
   if(!VSC_FUNC)
      return FUSEDMM_VSC_FAIL_RETURN;
   
   FP_AOP_FUNC AOP_FUNC = GetAOPFunc(GET_AOP_FLAG(imessage));
   if(!AOP_FUNC)
      return FUSEDMM_AOP_FAIL_RETURN;
   
/*
 *  Implementation 
 */
#if defined(PTTIME) && defined(LOAD_BALANCE)
   omp_set_num_threads(NTHREADS);
   #pragma omp parallel reduction(+:status)  
   {
      INDEXTYPE RowPerThd, tt;
      INDEXTYPE i, rowb, rowe;
      INDEXTYPE Mnnz = 0; /* non-zero count in M rows  */
      INDEXTYPE deg, cumRow, curRow;
      INDEXTYPE id = omp_get_thread_num();
      INDEXTYPE nthreads = omp_get_num_threads(); 
      VALUETYPE T[k]; /* temporary space to hold result of vector compute */
      
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
         #pragma omp parallel for schedule(dynamic) reduction(+:status)  
      #else
         #pragma omp parallel for schedule(static) reduction(+:status)  
      #endif
   #endif
      for (INDEXTYPE i = 0; i < m; i++)
#endif
      {
         //const VALUETYPE *Xi = x + i * ldx; 
         const VALUETYPE *lhs = x + i * ldx; 
         //VALUETYPE *Zi = z + i * ldz; 
         VALUETYPE *O = z + i * ldz; 
#ifndef LOAD_BALANCE
         VALUETYPE T[k]; /* temporary space to hold result of vector compute */
#endif
         for (INDEXTYPE j=pntrb[i]; j < pntre[i]; j++)
         {
            VALUETYPE scal, out; 
            const VALUETYPE *cT = T; /* where T is const */ 
            // assumming K is small enough to not overflow stack
            INDEXTYPE cid = indx[j];
            //const VALUETYPE *yj = y + cid * ldy; 
            const VALUETYPE *rhs = y + cid * ldy; 

            status += VOP_FUNC(k,lhs,k,rhs,k,T);
            status += ROP_FUNC(k,lhs,k,cT,scal);
            status += SOP_FUNC(scal, out);
            status += VSC_FUNC(k,T,out, k,T);
            status += AOP_FUNC(k, T, k, O);
         }
      }
#if defined(PTTIME) && defined(LOAD_BALANCE)
   }
#endif
   return status;
}

