#ifndef DIABLO
#define DIABLO 

#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_IJ_mv.h"



/* smooth inequality-constrained minimization problem
 * min_d E(d) s.t. g(d) >= 0
 * assumptions: 
 *   1. E is twice continuously differentiable
 *   2. g is once continuously differentiable
 *   3. g is twice continuously differentiable (if using a full-Newton solver)
 * */


/* --------- functions that are essential to have been implemented ------------ */

// initialize diablo
void * diablo_Init();

// finalize diablo, clear any needed memory
void diablo_Finalize(void * diabloState);


// obtain dofOffsets for the problem
// dofOffsets are assumed to be initialized to be an array of length two
// application owns the offsets and is responsible for memory management
//
// examples of dofOffsets for a problem where d is of (global) dimension 20
// one processor:
//   rank 0: dofOffsets[0] =  0; dofOffsets[1] = 19;
//
// two processors:
//   rank 0: dofOffsets[0] =  0; dofOffsets[1] =  9;
//   rank 1: dofOffsets[0] = 10; dofOffsets[1] = 19;    
HYPRE_BigInt * diablo_GetDofOffsetsx(void * diabloState);

HYPRE_BigInt * diablo_GetDofOffsetsy(void * diabloState);


// obtain constraintOffsets for the problem
// constraintOffsets are assumed to be initialized as an array of length two
// application owns the offsets and is responsible for memory management
//
// example of constraintOffsets for a problem where g is of (global) dimension 1
// one processor:
//   rank 0: constraintOffsets[0] = 0; constraintOffsets[1] = 0;
//
// two processors:
//   rank 0: constraintOffsets[0] = 0; constraintOffsets[1] = 0;
//   rank 1: constraintOffsets[0] = 1; constraintOffsets[1] = 0;
//HYPRE_BigInt * diablo_GetConstraintOffsets(void * diabloState);

// obtain constraintMask for the problem
// constraintMask is assumed to be initialized as an array of length 1 + constraintOffsets[1] - constraintOffsets[0]
// application owns the constraintMask and is responsible for memory management
//
// example of constraintMask for a problem where g is of (global) dimension 3
// wherein the first and third constraints are "true" constraints
// one processor:
//   rank 0: constraintMask[0] = 1; constraintMask[1] = 0; constraintMask[2] = 1;
//
// two processors:
//   rank 0: constraintMask[0] = 1; constraintMask[1] = 0;
//   rank 1: constraintMask[0] = 1;
//HYPRE_Int * diablo_GetConstraintMask(void * diabloState);
//void diablo_GetConstraintMask(void * diabloState, HYPRE_ParVector * constraintMask);

void diablo_FIJ(void * diabloState, HYPRE_ParVector * x, HYPRE_ParVector * y,
		HYPRE_IJVector * FIJ);

void diablo_F(void * diabloState, HYPRE_ParVector * x, HYPRE_ParVector * y, 
		HYPRE_ParVector * F);

void diablo_QIJ(void * diabloState, HYPRE_ParVector * x, HYPRE_ParVector * y,
		HYPRE_IJVector * QIJ);

void diablo_Q(void * diabloState, HYPRE_ParVector * x, HYPRE_ParVector * y, 
		HYPRE_ParVector * Q);


void diablo_DxF_IJ(void * diabloState, HYPRE_ParVector * x, HYPRE_ParVector * y, HYPRE_IJMatrix * dFdxIJ);

void diablo_DxF(void * diabloState, HYPRE_ParVector * x, HYPRE_ParVector * y, HYPRE_ParCSRMatrix * dFdxcsr);

void diablo_DyF_IJ(void * diabloState, HYPRE_ParVector * x, HYPRE_ParVector * y, HYPRE_IJMatrix * dFdyIJ);

void diablo_DyF(void * diabloState, HYPRE_ParVector * x, HYPRE_ParVector * y, HYPRE_ParCSRMatrix * dFdycsr);

void diablo_DxQ_IJ(void * diabloState, HYPRE_ParVector * x, HYPRE_ParVector * y, HYPRE_IJMatrix * dQdxIJ);

void diablo_DxQ(void * diabloState, HYPRE_ParVector * x, HYPRE_ParVector * y,
		HYPRE_ParCSRMatrix * dQdxcsr);

void diablo_DyQ_IJ(void * diabloState, HYPRE_ParVector * x, HYPRE_ParVector * y, 
		HYPRE_IJMatrix * dQdyIJ);

void diablo_DyQ(void * diabloState, HYPRE_ParVector * x, HYPRE_ParVector * y,
		HYPRE_ParCSRMatrix * dQdycsr);

void diablo_PrintProblemInfo(void * diabloState);


/* -------- non essential functions... only needed for the surrogate diablo -----------*/

/* application context */
typedef struct {
  HYPRE_BigInt * dofOffsetsx; // ....
  HYPRE_BigInt * dofOffsetsy;
  HYPRE_IJVector FIJ;
  HYPRE_IJVector QIJ;
  HYPRE_IJMatrix dFdxIJ;
  HYPRE_IJMatrix dFdyIJ;
  HYPRE_IJMatrix dQdxIJ;
  HYPRE_IJMatrix dQdyIJ;
 
  HYPRE_IJVector ulIJ;
  int nGlb;
  HYPRE_BigInt dimx;
  HYPRE_BigInt dimy;

  int * dofsx;
  int * dofsy;
} AppCtx;

#endif
