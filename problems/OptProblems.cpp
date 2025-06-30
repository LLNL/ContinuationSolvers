#include "mfem.hpp"
#include "OptProblems.hpp"



ParGeneralOptProblem::ParGeneralOptProblem() : block_offsetsx(3) { label = -1; }

void ParGeneralOptProblem::Init(HYPRE_BigInt * dofOffsetsU_, HYPRE_BigInt * dofOffsetsM_)
{
  dofOffsetsU = new HYPRE_BigInt[2];
  dofOffsetsM = new HYPRE_BigInt[2];
  for(int i = 0; i < 2; i++)
  {
    dofOffsetsU[i] = dofOffsetsU_[i];
    dofOffsetsM[i] = dofOffsetsM_[i];
  }
  dimU = dofOffsetsU[1] - dofOffsetsU[0];
  dimM = dofOffsetsM[1] - dofOffsetsM[0];
  dimC = dimM;
  
  block_offsetsx[0] = 0;
  block_offsetsx[1] = dimU;
  block_offsetsx[2] = dimM;
  block_offsetsx.PartialSum();

  MPI_Allreduce(&dimU, &dimUglb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&dimM, &dimMglb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}

double ParGeneralOptProblem::CalcObjective(const mfem::BlockVector &x)
{
  int eval_err; // throw away
  return CalcObjective(x, eval_err);
}

void ParGeneralOptProblem::CalcObjectiveGrad(const mfem::BlockVector &x, mfem::BlockVector &y)
{
   Duf(x, y.GetBlock(0));
   Dmf(x, y.GetBlock(1));
}

void ParGeneralOptProblem::c(const mfem::BlockVector &x, mfem::Vector &y)
{
  int eval_err; // throw-away
  return c(x, y, eval_err);
}

ParGeneralOptProblem::~ParGeneralOptProblem()
{
   block_offsetsx.DeleteAll();
}


// min E(d) s.t. g(d) >= 0
// min_(d,s) E(d) s.t. c(d,s) := g(d) - s = 0, s >= 0
ParOptProblem::ParOptProblem() : ParGeneralOptProblem()
{
}

void ParOptProblem::Init(HYPRE_BigInt * dofOffsetsU_, HYPRE_BigInt * dofOffsetsM_)
{
  dofOffsetsU = new HYPRE_BigInt[2];
  dofOffsetsM = new HYPRE_BigInt[2];
  for(int i = 0; i < 2; i++)
  {
    dofOffsetsU[i] = dofOffsetsU_[i];
    dofOffsetsM[i] = dofOffsetsM_[i];
  }

  dimU = dofOffsetsU[1] - dofOffsetsU[0];
  dimM = dofOffsetsM[1] - dofOffsetsM[0];
  dimC = dimM;
  
  block_offsetsx[0] = 0;
  block_offsetsx[1] = dimU;
  block_offsetsx[2] = dimM;
  block_offsetsx.PartialSum();

  MPI_Allreduce(&dimU, &dimUglb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&dimM, &dimMglb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
  ml.SetSize(dimM); ml = 0.0;
  mfem::Vector negIdentDiag(dimM);
  negIdentDiag = -1.0;
  Ih = GenerateHypreParMatrixFromDiagonal(dofOffsetsM, negIdentDiag);
}


double ParOptProblem::CalcObjective(const mfem::BlockVector &x, int & eval_err)
{ 
   return E(x.GetBlock(0), eval_err); 
}


void ParOptProblem::Duf(const mfem::BlockVector &x, mfem::Vector &y) { DdE(x.GetBlock(0), y); }

void ParOptProblem::Dmf(const mfem::BlockVector & /*x*/, mfem::Vector &y) { y = 0.0; }

mfem::HypreParMatrix * ParOptProblem::Duuf(const mfem::BlockVector &x) 
{ 
   return DddE(x.GetBlock(0)); 
}

mfem::HypreParMatrix * ParOptProblem::Dumf(const mfem::BlockVector &/*x*/) { return nullptr; }

mfem::HypreParMatrix * ParOptProblem::Dmuf(const mfem::BlockVector &/*x*/) { return nullptr; }

mfem::HypreParMatrix * ParOptProblem::Dmmf(const mfem::BlockVector &/*x*/) { return nullptr; }

void ParOptProblem::c(const mfem::BlockVector &x, mfem::Vector &y, int & eval_err) // c(u,m) = g(u) - m 
{
   g(x.GetBlock(0), y, eval_err);
   y.Add(-1.0, x.GetBlock(1));  
}


mfem::HypreParMatrix * ParOptProblem::Duc(const mfem::BlockVector &x) 
{ 
   return Ddg(x.GetBlock(0)); 
}

mfem::HypreParMatrix * ParOptProblem::Dmc(const mfem::BlockVector &/*x*/) 
{ 
   return Ih;
} 

ParOptProblem::~ParOptProblem() 
{
  delete[] dofOffsetsU;
  delete[] dofOffsetsM;
  delete Ih;
}






ReducedProblem::ReducedProblem(ParOptProblem * problem_, HYPRE_Int * constraintMask)
{
  problem = problem_;
  J = nullptr;
  P = nullptr;
  
  HYPRE_BigInt * dofOffsets = problem->GetDofOffsetsU();

  // given a constraint mask, lets update the constraintOffsets
  // from the original problem
  int nLocConstraints = 0;
  int nProblemConstraints = problem->GetDimM();
  for (int i = 0; i < nProblemConstraints; i++)
  {
    if (constraintMask[i] == 1)
    {
      nLocConstraints += 1;
    }
  }

  HYPRE_BigInt * constraintOffsets_reduced;
  constraintOffsets_reduced = offsetsFromLocalSizes(nLocConstraints);


  HYPRE_BigInt * constraintOffsets;
  constraintOffsets = offsetsFromLocalSizes(nProblemConstraints);
  
  P = GenerateProjector(constraintOffsets, constraintOffsets_reduced, constraintMask);

  Init(dofOffsets, constraintOffsets_reduced);
  delete[] constraintOffsets_reduced;
  delete[] constraintOffsets;
}

ReducedProblem::ReducedProblem(ParOptProblem * problem_, mfem::HypreParVector & constraintMask)
{
  problem = problem_;
  J = nullptr;
  P = nullptr;
  
  HYPRE_BigInt * dofOffsets = problem->GetDofOffsetsU();

  // given a constraint mask, lets update the constraintOffsets
  // from the original problem
  int nLocConstraints = 0;
  int nProblemConstraints = problem->GetDimM();
  for (int i = 0; i < nProblemConstraints; i++)
  {
    if (constraintMask[i] == 1)
    {
      nLocConstraints += 1;
    }
  }

  HYPRE_BigInt * constraintOffsets_reduced;
  constraintOffsets_reduced = offsetsFromLocalSizes(nLocConstraints);



  HYPRE_BigInt * constraintOffsets;
  constraintOffsets = offsetsFromLocalSizes(nProblemConstraints);
  
  P = GenerateProjector(constraintOffsets, constraintOffsets_reduced, constraintMask);

  Init(dofOffsets, constraintOffsets_reduced);
  delete[] constraintOffsets_reduced;
  delete[] constraintOffsets;
}

// energy objective E(d)
double ReducedProblem::E(const mfem::Vector &d, int & eval_err)
{
  return problem->E(d, eval_err);
}


// gradient of energy objective
void ReducedProblem::DdE(const mfem::Vector &d, mfem::Vector & gradE)
{
  problem->DdE(d, gradE);
}


mfem::HypreParMatrix * ReducedProblem::DddE(const mfem::Vector &d)
{
  return problem->DddE(d);
}

void ReducedProblem::g(const mfem::Vector &d, mfem::Vector &gd, int & eval_err)
{
  mfem::Vector gdfull(problem->GetDimM()); gdfull = 0.0;
  problem->g(d, gdfull, eval_err);
  P->Mult(gdfull, gd);
}


mfem::HypreParMatrix * ReducedProblem::Ddg(const mfem::Vector &d)
{
  mfem::HypreParMatrix * Jfull = problem->Ddg(d);
  if (J)
  {
    delete J; J = nullptr;
  }
  J = ParMult(P, Jfull, true);
  return J;
}

ReducedProblem::~ReducedProblem()
{
  delete P;
  if (J)
  {
    delete J;
  }
}


