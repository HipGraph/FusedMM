#ifndef FUSED_SDMM_SPMM_INTERNAL_H
#define FUSED_SDMM_SPMM_INTERNAL_H

#ifdef __cplusplus
   extern "C"
   {
#endif

/*
 * NOTE: 
 * This header file is meant for internal use to implement the general kernel,
 * not intended for the user. See fusedMM.h for function prototypes
 */

/*
 * Function pointer for user define functions 
 */
typedef int (*FP_VOP_UDEF_FUNC)(INDEXTYPE lhs_dim, const VALUETYPE *lhs, 
      INDEXTYPE rhs_dim, const VALUETYPE *rhs, INDEXTYPE out_dim, 
      VALUETYPE *out); 

typedef int (*FP_ROP_UDEF_FUNC)(INDEXTYPE lhs_dim, const VALUETYPE *lhs, 
      INDEXTYPE rhs_dim, const VALUETYPE *rhs, VALUETYPE *out); 

typedef int (*FP_SOP_UDEF_FUNC)(VALUETYPE val, VALUETYPE *out); 

typedef int (*FP_VSC_UDEF_FUNC)(INDEXTYPE rhs_dim, const VALUETYPE *rhs, 
      VALUETYPE scal, INDEXTYPE out_dim, VALUETYPE *out); 

typedef int (*FP_AOP_UDEF_FUNC)(INDEXTYPE rhs_dim, const VALUETYPE *rhs, 
      INDEXTYPE out_dim, VALUETYPE *out); 
/*
 * Function pointer for each stage of operations 
 */
typedef int (*FP_VOP_FUNC)(INDEXTYPE lhs_dim, const VALUETYPE *lhs, 
      INDEXTYPE rhs_dim, const VALUETYPE *rhs, INDEXTYPE out_dim, 
      VALUETYPE *out); 
typedef int (*FP_ROP_FUNC)(INDEXTYPE lhs_dim, const VALUETYPE *lhs, 
      INDEXTYPE rhs_dim, const VALUETYPE *rhs, VALUETYPE *out); 
typedef int (*FP_SOP_FUNC)(VALUETYPE val, VALUETYPE *out); 
typedef int (*FP_VSC_FUNC)(INDEXTYPE rhs_dim, const VALUETYPE *rhs, 
      VALUETYPE scal, INDEXTYPE out_dim, VALUETYPE *out); 
typedef int (*FP_AOP_FUNC)(INDEXTYPE rhs_dim, const VALUETYPE *rhs, INDEXTYPE out_dim, 
      VALUETYPE *out); 
/*
 * USER DEFINE FUNC IMPLEMENTATION 
 * DUMMY function, always return error when not implemented by user but used in
 * message using UDEF
 */
#ifndef VOP_UDEF_IMPL 
int VOP_UDEF_FUNC(INDEXTYPE lhs_dim, const VALUETYPE *lhs, INDEXTYPE rhs_dim, 
      const VALUETYPE *rhs, INDEXTYPE out_dim, VALUETYPE *out)
{
   return FUSEDMM_UNDEFINED_USER_FUNCTION;  
}
#endif
#ifndef ROP_UDEF_IMPL 
int ROP_UDEF_FUNC(INDEXTYPE lhs_dim, const VALUETYPE *lhs, INDEXTYPE rhs_dim, 
      const VALUETYPE *rhs, VALUETYPE *out) 
{
   return FUSEDMM_UNDEFINED_USER_FUNCTION;  
}
#endif

#ifndef SOP_UDEF_IMPL 
int SOP_UDEF_FUNC(VALUETYPE val, VALUETYPE *out) 
{
   return FUSEDMM_UNDEFINED_USER_FUNCTION;  
}
#endif
#ifndef VSC_UDEF_IMPL 
int VSC_UDEF_FUNC(INDEXTYPE rhs_dim, const VALUETYPE *rhs, VALUETYPE scal, 
      INDEXTYPE out_dim, VALUETYPE *out) 
{
   return FUSEDMM_UNDEFINED_USER_FUNCTION;  
}
#endif
#ifndef AOP_UDEF_IMPL /* func prototype */
int AOP_UDEF_FUNC(INDEXTYPE rhs_dim, const VALUETYPE *rhs, INDEXTYPE out_dim, 
      VALUETYPE *out) 
{
   return FUSEDMM_UNDEFINED_USER_FUNCTION;  
}
#endif

#ifdef __cplusplus
   } // extern "C"
#endif

#endif /* end of FUSED_SDMM_SPMM_INTERNAL_H */
