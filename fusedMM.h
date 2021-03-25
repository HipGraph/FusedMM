#ifndef FUSED_SDMM_SPMM_KERNEL_H
#define FUSED_SDMM_SPMM_KERNEL_H

#ifdef __cplusplus
   extern "C"
   {
#endif 
/*
 * Messages for different operations
 *    VOP message :        x0~xF
 *    ROP message :        x00~x0F
 *    SOP message :        x000~x00F
 *    VSC(MOP) message :   x0000~x000F
 *    AOP message :        x00000~x0000F
 * NOTE: use get/set macro to get or set any flags 
 */
/* VOP : vector vector operation */
#define VOP_NOOP 0x0          /* No operatiorn, skip the phase */ 
#define VOP_COPY_LHS 0x1      /* copy left side vector operand */ 
#define VOP_COPY_RHS 0x2      /* copy right side vector operand */ 
#define VOP_ADD  0x3          /* Vout = Vlhs + Vrhs */
#define VOP_SUBL  0x4         /* Vout = Vlhs - Vrhs */
#define VOP_SUBR  0x5         /* Vout = Vrhs - Vlhs */
#define VOP_MAX  0x6          /* Vout = max (Vlhs, Vrhs) */
#define VOP_MIN  0x7          /* Vout = max (Vlhs, Vrhs) */
#define VOP_UDEF  0xF         /* USER DEFINED FUNC */
#define VOP_CLEAR(bvec) (((bvec)>>4)<<4)  /* clearing last 4 bit */
#define VOP_MASK(bvec) ((bvec) & 0xF)
#define GET_VOP_FLAG(imsg) VOP_MASK(imsg)
#define SET_VOP_FLAG(imsg, vflag)  (imsg = (VOP_CLEAR(imsg) | vflag)) 

/* ROP : vector reduction operation */
#define ROP_NOOP 0x00         /* NO OP */
#define ROP_DOT 0x10          /* dot product */
#define ROP_ADD_LHS 0x20      /* sum of the lhs vector */
#define ROP_ADD_RHS 0x30      /* sum of  the rhs vector */
#define ROP_NORML 0x40        /* square sum of lhs vector */ 
#define ROP_NORMR 0x50        /* square sum of rhs vector */
#define ROP_UDEF 0xF0         /* USER DEFINED FUNC */ 
#define ROP_CLEAR(bvec) ((bvec) & (~((int32_t)0xF0)))  
#define ROP_MASK(bvec) ((bvec) & 0xF0)  
#define SET_ROP_FLAG(imsg, vflag)  (imsg = (ROP_CLEAR(imsg) | vflag)) 
#define GET_ROP_FLAG(imsg) ROP_MASK(imsg)

/* SOP : Scalar operation */
#define SOP_NOOP 0x000        /* NO OP */
#define SOP_COPY 0x100        /* Scalar copy */ 
#define SOP_UDEF 0xF00        /* USER DEFINE FUNC */ 
#define SOP_CLEAR(bvec) ((bvec) & (~((int32_t)0xF00)))  
#define SOP_MASK(bvec) ((bvec) & 0xF00)  
#define SET_SOP_FLAG(imsg, vflag)  (imsg = (SOP_CLEAR(imsg) | vflag)) 
#define GET_SOP_FLAG(imsg) SOP_MASK(imsg)

/* VSC/MOP: vector scaling */
#define VSC_NOOP 0x0000       /* NO OP */
#define VSC_MUL 0x1000        /* Vout[i] = scal * Vin[i] */
#define VSC_ADD 0x2000        /* Vout[i] = scal + Vin[i] */ 
#define VSC_UDEF 0xF000       /* USER DEFINE FUNC */ 
#define VSC_CLEAR(bvec) ((bvec) & (~((int32_t)0xF000)))  
#define VSC_MASK(bvec) ((bvec) & 0xF000)  
#define SET_VSC_FLAG(imsg, vflag)  (imsg = (VSC_CLEAR(imsg) | vflag)) 
#define GET_VSC_FLAG(imsg) VSC_MASK(imsg)

/* AOP : accumulate operation */
#define AOP_NOOP 0x00000      /* NO OP */
#define AOP_ADD 0x10000       /* Vout[i] += Vin[i]*/
#define AOP_MAX 0x20000       /* Vout[i] = max(Vout[i], vin[i]) */
#define AOP_MIN 0x30000       /* Vout[i] = min(Vout[i], vin[i]) */
#define AOP_UDEF 0xF0000      /* USER DEFINE FUNC */ 
#define AOP_CLEAR(bvec) ((bvec) & (~((int32_t)0xF0000)))  
#define AOP_MASK(bvec) ((bvec) & 0xF0000)  
#define SET_AOP_FLAG(imsg, vflag)  (imsg = (AOP_CLEAR(imsg) | vflag)) 
#define GET_AOP_FLAG(imsg) AOP_MASK(imsg)


int fusedMM_csr 
(
   const int32_t imessage,    /* message to dictate the operations */
   const INDEXTYPE m,         /* number of row of X */
   const INDEXTYPE n,         /* number of row of Y */
   const INDEXTYPE k,         /* feature dimension (col of X or Y) */
   const VALUETYPE alpha,     /* not used yet in general fusedMM */
   const INDEXTYPE nnz,       /* nonzeros in sparse matrix */
   const INDEXTYPE rows,      /* number of rows in sparse matrix */
   const INDEXTYPE cols,      /* number of columns in sparse matrix */
   const VALUETYPE *val,      /* value of non-zeros */
   const INDEXTYPE *indx,     /* colids -> column indices */
   const INDEXTYPE *pntrb,    /* starting of rowptr for each row: rowptr */
   const INDEXTYPE *pntre,    /* ending of rowptr for each row: rowptr+1 */
   const VALUETYPE *x,        /* Dense X matrix */
   const INDEXTYPE ldx,       /* 1eading dimension of X */
   const VALUETYPE *y,        /* Dense Y matrix */
   const INDEXTYPE ldy,       /* leading dimension of Y */
   const VALUETYPE beta,      /* beta value, Z = alpha*func(X,Y,A) + beta*Z */
   VALUETYPE *z,              /* Dense matrix Z */
   const INDEXTYPE ldz        /* leading dimension size of Z */
);

/*
 * Function prototype for user defined functions 
 */
/* return status of user defined functions */
#define FUSEDMM_SUCCESS_RETURN 0
#define FUSEDMM_FAIL_RETURN 1
#define FUSEDMM_VOP_FAIL_RETURN 2
#define FUSEDMM_ROP_FAIL_RETURN 4
#define FUSEDMM_AOP_FAIL_RETURN 8
#define FUSEDMM_SOP_FAIL_RETURN 16
#define FUSEDMM_VSC_FAIL_RETURN 32
#define FUSEDMM_NOT_ENOUGH_MEM -1
#define FUSEDMM_UNDEFINED_USER_FUNCTION 64 
#define FUSEDMM_NO_OPT_IMPL 128 

/*
 * USER DEFINE FUNC PROTOTYPE
 *    NOTE: define these macros when user provides the user defined functions
 *    For Example, enabling following three macros means, user will provide 
 *    user defined function for SOP(which is defined in 
 *    fusedMMtime.cpp). These user functions will be used when SOP_UDEF messages
 *    are used.
 *    Disable these macros if you don't have any user defined functions. 
 *    When these macros are not defined, but UDEF message is used the default
 *    UDEF function will return FUSEDMM_UNDEFINED_USER_FUNTION status.
 *   
 *    NOTE NOTE:  If user provides USER_DEFINE_FUNC, it must be thread safe. 
 *    User should not update any global varibale inside these functions!!!
 */

//#define VOP_UDEF_IMPL 1 
//#define ROP_UDEF_IMPL 1 
#define SOP_UDEF_IMPL 1 
//#define VSC_UDEF_IMPL 1 
//#define AOP_UDEF_IMPL 1 

int VOP_UDEF_FUNC(INDEXTYPE lhs_dim, const VALUETYPE *lhs, INDEXTYPE rhs_dim, 
      const VALUETYPE *rhs, INDEXTYPE out_dim, VALUETYPE *out); 
int ROP_UDEF_FUNC(INDEXTYPE lhs_dim, const VALUETYPE *lhs, INDEXTYPE rhs_dim, 
      const VALUETYPE *rhs, VALUETYPE *out); 
int SOP_UDEF_FUNC(VALUETYPE val, VALUETYPE *out); 
int VSC_UDEF_FUNC(INDEXTYPE rhs_dim, const VALUETYPE *rhs, VALUETYPE scal, 
      INDEXTYPE out_dim, VALUETYPE *out); 
int AOP_UDEF_FUNC(INDEXTYPE rhs_dim, const VALUETYPE *rhs, INDEXTYPE out_dim, 
      VALUETYPE *out); 

#ifdef __cplusplus
   }  // extern "C"
#endif

#endif /* end of FUSED_SDMM_SPMM_KERNEL_H */
