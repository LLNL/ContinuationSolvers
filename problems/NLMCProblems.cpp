#include "mfem.hpp"
#include "NLMCProblems.hpp"
#include "../utilities.hpp"



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

  xyoffsets.SetSize(3); 
  xyoffsets[0] = 0;
  xyoffsets[1] = dimx;
  xyoffsets[2] = dimy;
  xyoffsets.PartialSum(); 
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


OptNLMCProblem::OptNLMCProblem(OptProblem * optproblem_)
{
   optproblem = optproblem_;
   
   // x = dual variable
   // y = primal variable
   Init(optproblem->GetDofOffsetsM(), optproblem->GetDofOffsetsU());

   {
      mfem::Vector temp(dimx); temp = 0.0;
      dFdx = GenerateHypreParMatrixFromDiagonal(dofOffsetsx, temp);
   }
}

// F(x, y) = g(y)
void OptNLMCProblem::F(const mfem::Vector & x, const mfem::Vector & y, mfem::Vector & feval, int & eval_err, bool /*new_pt*/) const
{
  MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && feval.Size() == dimx, "OptNLMCProblem::F -- Inconsistent dimensions");
  optproblem->g(y, feval, eval_err);
}




// Q(x, y) = \nabla_y L(y, x) = \nabla_y E(y) - (dg(y)/ dy)^T x
void OptNLMCProblem::Q(const mfem::Vector & x, const mfem::Vector & y, mfem::Vector & qeval, int &eval_err, bool /*new_pt*/) const
{
  MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && qeval.Size() == dimy, "OptNLMCProblem::Q -- Inconsistent dimensions");
  
  optproblem->DdE(y, qeval);
  
  mfem::Operator * J = optproblem->Ddg(y);
  mfem::Vector temp(dimy); temp = 0.0;
  J->MultTranspose(x, temp);
  
  eval_err = 0;
  qeval.Add(-1.0, temp);
}


// dF/dx = 0
mfem::Operator * OptNLMCProblem::DxF(const mfem::Vector & /*x*/, const mfem::Vector & /*y*/, bool /*new_pt*/)
{
   return dFdx;
}

// dF/dy = dg/dy
mfem::Operator * OptNLMCProblem::DyF(const mfem::Vector & /*x*/, const mfem::Vector & y, bool /*new_pt*/)
{
   return optproblem->Ddg(y);
}


// dQ/dx = -(dg/dy)^T
mfem::Operator * OptNLMCProblem::DxQ(const mfem::Vector & /*x*/, const mfem::Vector & y, bool /*new_pt*/)
{
   mfem::Operator * J = optproblem->Ddg(y);
   auto Jhypre = dynamic_cast<mfem::HypreParMatrix *>(J);
   MFEM_VERIFY(Jhypre, "expecting a HypreParMatrix Ddg");
   if (dQdx)
   {
      delete dQdx;
      dQdx = nullptr;
   }
   dQdx = Jhypre->Transpose();
   mfem::Vector temp(dimy); temp = -1.0;
   dQdx->ScaleRows(temp);
   return dQdx;
}


// dQdy = Hessian(E) - second order derivaives in g
mfem::Operator * OptNLMCProblem::DyQ(const mfem::Vector & /*x*/, const mfem::Vector & y, bool /*new_pt*/)
{
   return optproblem->DddE(y);
}


OptNLMCProblem::~OptNLMCProblem()
{
   if (dFdx)
   {
      delete dFdx;
   }
   if (dQdx)
   {
      delete dQdx;
   }
};

EqualityConstrainedHomotopyProblem::EqualityConstrainedHomotopyProblem()
{
   y_partition.SetSize(3);
   adjoint_solver = new DirectSolver();
};

mfem::Vector EqualityConstrainedHomotopyProblem::GetDisplacement(mfem::Vector &X)
{
   MFEM_VERIFY(X.Size() == dimx + dimy, "input vector of an invalid size");   
   mfem::Vector u(X, 0, dimu);
   return u;
};

mfem::Vector EqualityConstrainedHomotopyProblem::GetLagrangeMultiplier(mfem::Vector &X)
{
   MFEM_VERIFY(X.Size() == dimx + dimy, "input vector of an invalid size");   
   mfem::Vector multiplier(X, dimu, dimc);
   return multiplier;
};


void EqualityConstrainedHomotopyProblem::SetSizes(HYPRE_BigInt * uOffsets, HYPRE_BigInt * cOffsets)
{
   set_sizes = true;
   dimu = uOffsets[1] - uOffsets[0];
   dimc = cOffsets[1] - cOffsets[0];
   uOffsets_ = new HYPRE_BigInt[2];
   cOffsets_ = new HYPRE_BigInt[2];
   for (int i = 0; i < 2; i++)
   {
      uOffsets_[i] = uOffsets[i];
      cOffsets_[i] = cOffsets[i];
   }
   
   y_partition[0] = 0;
   y_partition[1] = dimu;
   y_partition[2] = dimc;
   y_partition.PartialSum();

   HYPRE_BigInt dofOffsets[2];
   HYPRE_BigInt complementarityOffsets[2];
   for (int i = 0; i < 2; i++) {
      dofOffsets[i] = uOffsets_[i] + cOffsets_[i];
      complementarityOffsets[i] = 0;
   }
   Init(complementarityOffsets, dofOffsets);
   
   // dF / dx 0 x 0 matrix
   {
     int nentries = 0;
     auto temp = new mfem::SparseMatrix(dimx, dimxglb, nentries);
     dFdx = GenerateHypreParMatrixFromSparseMatrix(dofOffsetsx, dofOffsetsx, temp);
     delete temp;
   }

   // dF / dy 0 x dimy matrix
   {
     int nentries = 0;
     auto temp = new mfem::SparseMatrix(dimx, dimyglb, nentries);
     dFdy = GenerateHypreParMatrixFromSparseMatrix(dofOffsetsx, dofOffsetsy, temp);
     delete temp;
   }

   // dQ / dx dimy x 0 matrix
   {
     int nentries = 0;
     auto temp = new mfem::SparseMatrix(dimy, dimxglb, nentries);
     dQdx = GenerateHypreParMatrixFromSparseMatrix(dofOffsetsy, dofOffsetsx, temp);
     delete temp;
   }
   q_cache.SetSize(dimy); q_cache = 0.0;
};

void EqualityConstrainedHomotopyProblem::F(const mfem::Vector& x, const mfem::Vector& y, mfem::Vector& feval, int& Feval_err, bool /*new_pt*/) const
{
   MFEM_VERIFY(set_sizes, "need to set sizes in problem constructor");
   MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && feval.Size() == dimx,
              "F -- Inconsistent dimensions");
  feval = 0.0;
  Feval_err = 0;
};

// Q = [  r + (dc/du)^T l]
//     [ -c ]
void EqualityConstrainedHomotopyProblem::Q(const mfem::Vector& x, const mfem::Vector& y, mfem::Vector& qeval, int& Qeval_err, bool new_pt) const
{
  MFEM_VERIFY(set_sizes, "need to set sizes in problem constructor");
  MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && qeval.Size() == dimy,
              "Q -- Inconsistent dimensions");
  
  if (new_pt)
  {
     qeval = 0.0;
     mfem::BlockVector yblock(y_partition);
     yblock.Set(1.0, y);
     mfem::BlockVector qblock(y_partition);
     qblock = 0.0;

     auto u = yblock.GetBlock(0);
     auto l = yblock.GetBlock(1);
     auto residual_vector = residual(u, new_pt);
     qblock.GetBlock(0).Set(1.0, residual_vector);
     auto residual_contribution = constraintJacobianTvp(u, l, new_pt);
     qblock.GetBlock(0).Add(1.0, residual_contribution);

     auto constraint_eval = constraint(u, new_pt);
     qblock.GetBlock(1).Set(-1.0, constraint_eval);

     qeval.Set(1.0, qblock);
     q_cache.Set(1.0, qeval);
  }
  else
  {
     qeval.Set(1.0, q_cache);
  }
  Qeval_err = 0;
  int Qeval_err_loc = 0;
  for (int i = 0; i < qeval.Size(); i++) {
    if (std::isnan(qeval(i))) {
      Qeval_err_loc = 1;
      break;
    }
  }
  MPI_Allreduce(&Qeval_err_loc, &Qeval_err, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
};





// dQdy = [ dr/du   dc/du^T]
//        [-dc/du   0  ]
mfem::Operator* EqualityConstrainedHomotopyProblem::DyQ(const mfem::Vector& /*x*/, const mfem::Vector& y, bool new_pt)
{
  MFEM_VERIFY(set_sizes, "need to set sizes in problem constructor");
  MFEM_VERIFY(y.Size() == dimy, "InertialReliefProblem::DyQ -- Inconsistent dimensions");
  if (new_pt)
  {
     // note we are neglecting Hessian constraint terms
     mfem::BlockVector yblock(y_partition);
     yblock.Set(1.0, y);
     auto u = yblock.GetBlock(0);

     if (dQdy) {
       delete dQdy;
     }
     {
       auto drdu = residualJacobian(u, new_pt);
       auto negdcdu = constraintJacobian(u, new_pt);
       auto dcduT = negdcdu->Transpose();
       (*negdcdu) *= -1.0;

       mfem::Array2D<const mfem::HypreParMatrix*> BlockMat(2, 2);
       BlockMat(0, 0) = drdu;
       BlockMat(0, 1) = dcduT;
       BlockMat(1, 0) = negdcdu;
       BlockMat(1, 1) = nullptr;
       dQdy = HypreParMatrixFromBlocks(BlockMat);
       delete dcduT;
     }
  }
  return dQdy;
};

void EqualityConstrainedHomotopyProblem::SetAdjointSolver(mfem::Solver * adjoint_solver_)
{
   own_adjoint_solver = false;
   adjoint_solver = adjoint_solver_;
};


// evaluation_u_point: point at which adjoint system will be evaluated
// adjoint_load: rhs forcing term of the adjoint equation, determined by 
//               design objective, etc.
// adjoint: solution of the adjoint equation
void EqualityConstrainedHomotopyProblem::AdjointSolve(const mfem::Vector & evaluation_u_point, const mfem::Vector & adjoint_load, 
   mfem::Vector & adjoint)
{
   MFEM_VERIFY(adjoint_load.Size() == dimu + dimc, "Adjoint load not of the correct size");
   MFEM_VERIFY(adjoint.Size() == dimu + dimc, "Adjoint solution vector not of the correct size");
   mfem::BlockVector evaluation_y_point(y_partition); evaluation_y_point = 0.0;
   evaluation_y_point.GetBlock(0).Set(1.0, evaluation_u_point);
   mfem::Vector evaluation_x_point(dimx); evaluation_x_point = 0.0;
   auto A = DyQ(evaluation_x_point, evaluation_y_point);
   if (adjoint_is_symmetric)
   {
      adjoint_solver->SetOperator(*A);
      adjoint_solver->Mult(adjoint_load, adjoint);
   }
   else
   {
      auto Ahypre = dynamic_cast<mfem::HypreParMatrix*>(A);
      auto Aadjoint = Ahypre->Transpose();
      
      adjoint_solver->SetOperator(*Aadjoint);
      adjoint_solver->Mult(adjoint_load, adjoint);
      delete Aadjoint;
   }
};


EqualityConstrainedHomotopyProblem::~EqualityConstrainedHomotopyProblem()
{
  if (set_sizes)
  {
     delete[] uOffsets_;
     delete[] cOffsets_;
     delete dFdx;
     delete dFdy;
     delete dQdx;
  }
  if (dQdy)
  {
     delete dQdy;
  }
  if (own_adjoint_solver)
  {
     delete adjoint_solver;
  }
};

