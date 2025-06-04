#include <stdio.h>
#include <stdlib.h>
#include "diablo.h"


/* generate a random floating point number from min to max */
double randfrom(double min, double max) 
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

/* diablo surrogate */
/* NLMC Target problem
 * 0 <= x \perp F(x, y) >= 0
 *              Q(x, y)  = 0
 * with F(x, y) = y - ul
 *      Q(x, y) = y - x
 * which corresponds to the
 * first-order optimality conditions
 * for the convex quadratic programming problem
 * min_u E(u) = u^T u / 2
 * s.t. u - u_l >=0
 * where x is the Lagrange multiplier associated
 * to the optimization problem and 
 * y = u, the primal variable
 * 
 * this library contains functionality to 
 * 1. evaluate F(x, y) and its Jacobians dF/dx, dF/dy
 * 2. evaluate Q(x, y) and its Jacobians dQ/dx, dQ/dy
 *
 * x will be partitioned among MPI processes
 * according to dofOffsetsx that is the indicies
 * of the global vector x
 * dofOffsetsx[0] <= i <= dofOffsetsx[1] are owned by the 
 * given process
 *
 * y will be partitioned among MPI processes
 * according to dofOffsetsy that is the indicies 
 * of the global vector y 
 * dofOffsetsy[0] <= i <= dofOffsetsy[1] are owned
 * by the given process
 *
 * F will be partitioned like x
 * Q will be partitioned like y
 *
 * it is assumed that x n-dimensional and g(x) is m-dimensional
 * where m <= n. A technicality is that we will have
 * dofOffsets = constraintOffsets which means, that effectively,
 * m = n. However, we use a constraintMask, a binary 0,1 
 * n-vector which provides information about which entries of g
 * are true constraints that we would like to expose to the 
 * numerical optimizer
 * */



void * diablo_Init()
{
  int nGlb = 20;
  AppCtx * ctx = (AppCtx *) calloc(1, sizeof(AppCtx));
  ctx->nGlb = nGlb;
  int myrank, nprocs; 
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  ctx->dofOffsetsx = (HYPRE_BigInt *) calloc(2, sizeof(HYPRE_BigInt));
  ctx->dofOffsetsy = (HYPRE_BigInt *) calloc(2, sizeof(HYPRE_BigInt));
 
  for (int i = 0; i < 2; i++)
  {
    (ctx->dofOffsetsx)[i] = (HYPRE_Int) ((myrank + i ) * nGlb / nprocs);
  }
  ctx->dofOffsetsx[1] -= 1;
  
  for (int i = 0; i < 2; i++)
  {
    ctx->dofOffsetsy[i] = ctx->dofOffsetsx[i];
  }

  ctx->dimx = 1 + (ctx->dofOffsetsx[1] - ctx->dofOffsetsx[0]);
  ctx->dimy = 1 + (ctx->dofOffsetsy[1] - ctx->dofOffsetsy[0]);
  
  ctx->dofsx = (int *) calloc(ctx->dimx, sizeof(int));
  for (int i = 0; i < ctx->dimx; i++)
  {
    ctx->dofsx[i] = ctx->dofOffsetsx[0] + i;
  }
  
  ctx->dofsy = (int *) calloc(ctx->dimy, sizeof(int));
  for (int i = 0; i < ctx->dimy; i++)
  {
    ctx->dofsy[i] = ctx->dofOffsetsy[0] + i;
  }

  printf("dimy = %d\n", ctx->dimy);
  printf("y offsets = %d, %d\n", ctx->dofOffsetsy[0], ctx->dofOffsetsy[1]);
  
  
  HYPRE_IJVectorCreate(MPI_COMM_WORLD, ctx->dofOffsetsy[0], ctx->dofOffsetsy[1], &(ctx->ulIJ));
  HYPRE_IJVectorSetObjectType(ctx->ulIJ, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(ctx->ulIJ);
  double * valuesul = (double *) calloc(ctx->dimy, sizeof(double));
  for (int i = 0; i < ctx->dimy; i++)
  {
     valuesul[i] = randfrom(-1.0, 1.0);
  } 

  HYPRE_IJVectorSetValues((ctx->ulIJ), ctx->dimy, ctx->dofsy, valuesul);
  HYPRE_IJVectorAssemble((ctx->ulIJ));
  free(valuesul);
  

  
  
  
  HYPRE_IJVectorCreate(MPI_COMM_WORLD, ctx->dofOffsetsx[0], ctx->dofOffsetsx[1], &(ctx->FIJ));
  HYPRE_IJVectorSetObjectType(ctx->FIJ, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(ctx->FIJ);
  
  HYPRE_IJVectorCreate(MPI_COMM_WORLD, ctx->dofOffsetsy[0], ctx->dofOffsetsy[1], &(ctx->QIJ));
  HYPRE_IJVectorSetObjectType(ctx->QIJ, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(ctx->QIJ);


  HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ctx->dofOffsetsx[0], ctx->dofOffsetsx[1],
		                       ctx->dofOffsetsx[0], ctx->dofOffsetsx[1], &(ctx->dFdxIJ));
  HYPRE_IJMatrixSetObjectType(ctx->dFdxIJ, HYPRE_PARCSR);
  HYPRE_IJMatrixInitialize(ctx->dFdxIJ);
  
  HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ctx->dofOffsetsx[0], ctx->dofOffsetsx[1],
		                       ctx->dofOffsetsy[0], ctx->dofOffsetsy[1], &(ctx->dFdyIJ));
  HYPRE_IJMatrixSetObjectType(ctx->dFdyIJ, HYPRE_PARCSR);
  HYPRE_IJMatrixInitialize(ctx->dFdyIJ);
  
  HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ctx->dofOffsetsy[0], ctx->dofOffsetsy[1],
		                       ctx->dofOffsetsx[0], ctx->dofOffsetsx[1], &(ctx->dQdxIJ));
  HYPRE_IJMatrixSetObjectType(ctx->dQdxIJ, HYPRE_PARCSR);
  HYPRE_IJMatrixInitialize(ctx->dQdxIJ);
  
  HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ctx->dofOffsetsy[0], ctx->dofOffsetsy[1],
		                       ctx->dofOffsetsy[0], ctx->dofOffsetsy[1], &(ctx->dQdyIJ));
  HYPRE_IJMatrixSetObjectType(ctx->dQdyIJ, HYPRE_PARCSR);
  HYPRE_IJMatrixInitialize(ctx->dQdyIJ);

  return ctx;
}

void diablo_Finalize(void * diabloState)
{
  AppCtx * ctx = (AppCtx *) diabloState;
  free((ctx->dofOffsetsx));
  free((ctx->dofOffsetsy));
  free((ctx->dofsx));
  free((ctx->dofsy));
  HYPRE_IJVectorDestroy(ctx->FIJ);
  HYPRE_IJVectorDestroy(ctx->QIJ);
  HYPRE_IJVectorDestroy(ctx->ulIJ);
  HYPRE_IJMatrixDestroy(ctx->dFdxIJ);
  HYPRE_IJMatrixDestroy(ctx->dFdyIJ);
  HYPRE_IJMatrixDestroy(ctx->dQdxIJ);
  HYPRE_IJMatrixDestroy(ctx->dQdyIJ);
  
  free(ctx);
}


HYPRE_BigInt * diablo_GetDofOffsetsx(void * diabloState)
{
  AppCtx * ctx = (AppCtx *) diabloState;

  HYPRE_BigInt * dofOffsetsx = (HYPRE_BigInt *) calloc(2, sizeof(HYPRE_BigInt));
  for(int i = 0; i < 2; i++)
  {
    dofOffsetsx[i] = (ctx->dofOffsetsx)[i];
  }
  return dofOffsetsx;
}

HYPRE_BigInt * diablo_GetDofOffsetsy(void * diabloState)
{
  AppCtx * ctx = (AppCtx *) diabloState;

  HYPRE_BigInt * dofOffsetsy = (HYPRE_BigInt *) calloc(2, sizeof(HYPRE_BigInt));
  for(int i = 0; i < 2; i++)
  {
    dofOffsetsy[i] = (ctx->dofOffsetsy)[i];
  }
  return dofOffsetsy;

}


// can we clean this up with Add functions? 
void diablo_FIJ(void * diabloState, HYPRE_ParVector * x, HYPRE_ParVector * y, HYPRE_IJVector * FIJ)
{
   AppCtx * ctx = (AppCtx *) diabloState;
   double * valuesy = (double *) calloc(ctx->dimy, sizeof(double));
   HYPRE_ParVectorGetValues(*y, ctx->dimy, ctx->dofsy, valuesy);

   double * valuesul = (double *) calloc(ctx->dimy, sizeof(double));
   HYPRE_IJVectorGetValues(ctx->ulIJ, ctx->dimy, ctx->dofsy, valuesul);
   double * valuesF = (double *) calloc(ctx->dimx, sizeof(double));
   for (int i = 0; i < ctx->dimx; i++)
   {
     valuesF[i] = valuesy[i] - valuesul[i]; // - valuesul[i]; // TO DO: include - ul term
   }

   HYPRE_IJVectorSetValues(*FIJ, ctx->dimx, ctx->dofsx, valuesF);
   HYPRE_IJVectorAssemble(*FIJ);

   free(valuesul);
   free(valuesy);
   free(valuesF);
}


void diablo_F(void * diabloState, HYPRE_ParVector * x, HYPRE_ParVector * y, HYPRE_ParVector * F)
{
   AppCtx * ctx = (AppCtx *) diabloState;
   diablo_FIJ(diabloState, x, y, &(ctx->FIJ));
   HYPRE_IJVectorGetObject(ctx->FIJ, (void **) F);
}


// to do: clean this up: potentially use some axpy functions
void diablo_QIJ(void * diabloState, HYPRE_ParVector * x, HYPRE_ParVector * y, HYPRE_IJVector * QIJ)
{
   AppCtx * ctx = (AppCtx *) diabloState;
   
   double * valuesy = (double *) calloc(ctx->dimy, sizeof(double));
   HYPRE_ParVectorGetValues(*y, ctx->dimy, ctx->dofsy, valuesy);

   double * valuesx = (double *) calloc(ctx->dimx, sizeof(double));
   HYPRE_ParVectorGetValues(*x, ctx->dimx, ctx->dofsx, valuesx);
   


   double * valuesQ = (double *) calloc(ctx->dimy, sizeof(double));
   for (int i = 0; i < ctx->dimy; i++)
   {
      valuesQ[i] = valuesy[i] - valuesx[i];
   }

   HYPRE_IJVectorSetValues(*QIJ, ctx->dimy, ctx->dofsy, valuesQ);
   HYPRE_IJVectorAssemble(*QIJ);

   free(valuesx);
   free(valuesy);
   free(valuesQ);
}

void diablo_Q(void * diabloState, HYPRE_ParVector * x, HYPRE_ParVector * y, HYPRE_ParVector * Q)
{
   AppCtx * ctx = (AppCtx *) diabloState;
   diablo_QIJ(diabloState, x, y, &(ctx->QIJ));
   HYPRE_IJVectorGetObject(ctx->QIJ, (void **) Q);
}


void diablo_DxF_IJ(void * diabloState, HYPRE_ParVector * x, HYPRE_ParVector * y, HYPRE_IJMatrix * dFdxIJ)
{
   AppCtx * ctx = (AppCtx *) diabloState;
   int nnz = 1; // nonzeros per row
   int * cols = (int *) calloc(nnz, sizeof(int));
   double * values = (double *) calloc(nnz, sizeof(double)); values[0] = 0.0;
   for (int i = ctx->dofOffsetsx[0]; i <= ctx->dofOffsetsx[1]; i++)
   {
      cols[0] = i;
      HYPRE_IJMatrixSetValues(*dFdxIJ, 1, &nnz, &i, cols, values);
   }
   HYPRE_IJMatrixAssemble(*dFdxIJ);
   free(cols);
   free(values);
}


void diablo_DxF(void * diabloState, HYPRE_ParVector * x, HYPRE_ParVector * y, HYPRE_ParCSRMatrix * dFdxcsr)
{
  AppCtx * ctx = (AppCtx *) diabloState;
  diablo_DxF_IJ(diabloState, x, y, &(ctx->dFdxIJ));
  HYPRE_IJMatrixGetObject(ctx->dFdxIJ, (void**) dFdxcsr);
}

void diablo_DyF_IJ(void * diabloState, HYPRE_ParVector * x, HYPRE_ParVector * y, HYPRE_IJMatrix * dFdyIJ)
{
   AppCtx * ctx = (AppCtx *) diabloState;
   int nnz = 1; // nonzeros per row
   int * cols = (int *) calloc(nnz, sizeof(int));
   double * values = (double *) calloc(nnz, sizeof(double)); values[0] = 1.0;
   for (int i = ctx->dofOffsetsx[0]; i <= ctx->dofOffsetsx[1]; i++)
   {
      cols[0] = i;
      HYPRE_IJMatrixSetValues(*dFdyIJ, 1, &nnz, &i, cols, values);
   }
   HYPRE_IJMatrixAssemble(*dFdyIJ);
   free(cols);
   free(values);
}


void diablo_DyF(void * diabloState, HYPRE_ParVector * x, HYPRE_ParVector * y, HYPRE_ParCSRMatrix * dFdycsr)
{
  AppCtx * ctx = (AppCtx *) diabloState;
  diablo_DyF_IJ(diabloState, x, y, &(ctx->dFdyIJ));
  HYPRE_IJMatrixGetObject(ctx->dFdyIJ, (void**) dFdycsr);
}



void diablo_DxQ_IJ(void * diabloState, HYPRE_ParVector * x, HYPRE_ParVector * y, HYPRE_IJMatrix * dQdxIJ)
{
   AppCtx * ctx = (AppCtx *) diabloState;
   int nnz = 1; // nonzeros per row
   int * cols = (int *) calloc(nnz, sizeof(int));
   double * values = (double *) calloc(nnz, sizeof(double)); values[0] = -1.0;
   for (int i = ctx->dofOffsetsy[0]; i <= ctx->dofOffsetsy[1]; i++)
   {
      cols[0] = i;
      HYPRE_IJMatrixSetValues(*dQdxIJ, 1, &nnz, &i, cols, values);
   }
   HYPRE_IJMatrixAssemble(*dQdxIJ);
   free(cols);
   free(values);
}


void diablo_DxQ(void * diabloState, HYPRE_ParVector * x, HYPRE_ParVector * y, HYPRE_ParCSRMatrix * dQdxcsr)
{
  AppCtx * ctx = (AppCtx *) diabloState;
  diablo_DxQ_IJ(diabloState, x, y, &(ctx->dQdxIJ));
  HYPRE_IJMatrixGetObject(ctx->dQdxIJ, (void**) dQdxcsr);
}

void diablo_DyQ_IJ(void * diabloState, HYPRE_ParVector * x, HYPRE_ParVector * y, HYPRE_IJMatrix * dQdyIJ)
{
   AppCtx * ctx = (AppCtx *) diabloState;
   int nnz = 1; // nonzeros per row
   int * cols = (int *) calloc(nnz, sizeof(int));
   double * values = (double *) calloc(nnz, sizeof(double)); values[0] = 1.0;
   for (int i = ctx->dofOffsetsy[0]; i <= ctx->dofOffsetsy[1]; i++)
   {
      cols[0] = i;
      HYPRE_IJMatrixSetValues(*dQdyIJ, 1, &nnz, &i, cols, values);
   }
   HYPRE_IJMatrixAssemble(*dQdyIJ);
   free(cols);
   free(values);
}


void diablo_DyQ(void * diabloState, HYPRE_ParVector * x, HYPRE_ParVector * y, HYPRE_ParCSRMatrix * dQdycsr)
{
  AppCtx * ctx = (AppCtx *) diabloState;
  diablo_DyQ_IJ(diabloState, x, y, &(ctx->dQdyIJ));
  HYPRE_IJMatrixGetObject(ctx->dQdyIJ, (void**) dQdycsr);
}


void diablo_PrintProblemInfo(void * diabloState)
{
   AppCtx * ctx = (AppCtx *) diabloState;
   double * valuesul = (double *) calloc(ctx->dimy, sizeof(double));
   HYPRE_IJVectorGetValues(ctx->ulIJ, ctx->dimy, ctx->dofsy, valuesul);

   for (int i = 0; i < ctx->dimy; i++)
   {
     printf("ul(%d) = %1.3e\n", i, valuesul[i]);
   }
   free(valuesul);
}
