#include "mfem.hpp"
#include "NLMCProblems.hpp"
#include "../solvers/Utilities.hpp"



GeneralNLMCProblem::GeneralNLMCProblem() 
{ 
  dofOffsetsx = nullptr;
  dofOffsetsy = nullptr;
  label = -1;
}

void GeneralNLMCProblem::Init(HYPRE_BigInt * dofOffsetsx_, HYPRE_BigInt * dofOffsetsy_)
{
  dofOffsetsx = new HYPRE_BigInt[2];
  dofOffsetsy = new HYPRE_BigInt[2];
  for(int i = 0; i < 2; i++)
  {
    dofOffsetsx[i] = dofOffsetsx_[i];
    dofOffsetsy[i] = dofOffsetsy_[i];
  }
  dimx = dofOffsetsx[1] - dofOffsetsx[0];
  dimy = dofOffsetsy[1] - dofOffsetsy[0];
  
  MPI_Allreduce(&dimx, &dimxglb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&dimy, &dimyglb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}



GeneralNLMCProblem::~GeneralNLMCProblem() 
{ 
   if (dofOffsetsx != nullptr)
   {
      delete[] dofOffsetsx;
   }
   if (dofOffsetsy != nullptr)
   {
      delete[] dofOffsetsy;
   }
}


// ------------------------------------


OptNLMCProblem::OptNLMCProblem(ParOptProblem * optproblem_)
{
   optproblem = optproblem_;
   
   // x = dual variable
   // y = primal variable
   Init(optproblem->GetDofOffsetsM(), optproblem->GetDofOffsetsU());

   {
      mfem::Vector temp(dimx); temp = 0.0;
      dFdx = GenerateHypreParMatrixFromDiagonal(dofOffsetsx, temp);
   }
   dFdy = nullptr;
   dQdx = nullptr;
   dQdy = nullptr; 
   Pc = nullptr;
}

// F(x, y) = g(y)
void OptNLMCProblem::F(const mfem::Vector & x, const mfem::Vector & y, mfem::Vector & feval, int & eval_err) const
{
  MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && feval.Size() == dimx, "OptNLMCProblem::F -- Inconsistent dimensions");
  optproblem->g(y, feval, eval_err);
}




// Q(x, y) = \nabla_y L(y, x) = \nabla_y E(y) - (dg(y)/ dy)^T x
void OptNLMCProblem::Q(const mfem::Vector & x, const mfem::Vector & y, mfem::Vector & qeval, int &eval_err) const
{
  MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && qeval.Size() == dimy, "OptNLMCProblem::Q -- Inconsistent dimensions");
  
  optproblem->DdE(y, qeval);
  
  mfem::HypreParMatrix * J = optproblem->Ddg(y);
  mfem::Vector temp(dimy); temp = 0.0;
  J->MultTranspose(x, temp);
  
  eval_err = 0;
  qeval.Add(-1.0, temp);
}


// dF/dx = 0
mfem::HypreParMatrix * OptNLMCProblem::DxF(const mfem::Vector & /*x*/, const mfem::Vector & /*y*/)
{
   return dFdx;
}

// dF/dy = dg/dy
mfem::HypreParMatrix * OptNLMCProblem::DyF(const mfem::Vector & /*x*/, const mfem::Vector & y)
{
   dFdy = optproblem->Ddg(y);
   return dFdy;
}


// dQ/dx = -(dg/dy)^T
mfem::HypreParMatrix * OptNLMCProblem::DxQ(const mfem::Vector & /*x*/, const mfem::Vector & y)
{
   mfem::HypreParMatrix * J = optproblem->Ddg(y);
   if (dQdx)
   {
      delete dQdx;
      dQdx = nullptr;
   }
   dQdx = J->Transpose();
   mfem::Vector temp(dimy); temp = -1.0;
   dQdx->ScaleRows(temp);
   return dQdx;
}


// dQdy = Hessian(E) - second order derivaives in g
mfem::HypreParMatrix * OptNLMCProblem::DyQ(const mfem::Vector & /*x*/, const mfem::Vector & y)
{
   return optproblem->DddE(y);
}


mfem::HypreParMatrix * OptNLMCProblem::GetRestrictionToConstrainedDofs()
{
   if (!Pc)
   {
      // TODO: could simply use dQdx instead of dFdy?
      MFEM_ASSERT(dFdy, "dFdy has not been formed!");
      Pc = NonZeroColMap(*dFdy);
   }

   return Pc;
}


OptNLMCProblem::~OptNLMCProblem()
{
   delete dFdx;
   if (dQdx)
   {
      delete dQdx;
   }
}

