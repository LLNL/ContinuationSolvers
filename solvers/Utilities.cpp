
#include "Utilities.hpp"


DirectSolver::DirectSolver()
   : mfem::Solver(), solver(nullptr)
{ }

DirectSolver::DirectSolver(const mfem::Operator& op)
   : DirectSolver()
{
   SetOperator(op);
}

DirectSolver::~DirectSolver()
{
   delete solver;
#ifdef MFEM_USE_STRUMPACK
   delete Astrumpack;
#endif
}

void DirectSolver::SetOperator(const mfem::Operator& op)
{
   height = op.NumRows();
   width = op.NumCols();

   auto op_ptr = dynamic_cast<const mfem::HypreParMatrix *>(&op);
   MFEM_VERIFY(op_ptr, "op must be a mfem::HypreParMatrix!");

#ifdef MFEM_USE_STRUMPACK
   auto directSolver = new mfem::STRUMPACKSolver(op_ptr->GetComm());
   directSolver->SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
   directSolver->SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);
   Astrumpack = new mfem::STRUMPACKRowLocMatrix(*op_ptr);
   directSolver->SetOperator(*Astrumpack);
#elif defined(MFEM_USE_MUMPS)
   auto directSolver = new mfem::MUMPSSolver(op_ptr->GetComm());
   directSolver->SetPrintLevel(0);
   directSolver->SetMatrixSymType(mfem::MUMPSSolver::MatType::UNSYMMETRIC);
   directSolver->SetOperator(*op_ptr);
#elif defined(MFEM_USE_MKL_CPARDISO)
   auto directSolver = new mfem::CPardisoSolver(op_ptr->GetComm());
   directSolver->SetOperator(*op_ptr);
#else
   MFEM_ABORT("DirectSolver will not work unless compiled mfem is with MUMPS, MKL_CPARDISO, or STRUMPACK");
#endif
   solver = directSolver;
}

void DirectSolver::Mult(const mfem::Vector &b, mfem::Vector &x) const 
{
   MFEM_VERIFY(solver, "SetOperator must be called before Mult!");
   solver->Mult(b, x);
}

mfem::HypreParMatrix * NonZeroRowMap(const mfem::HypreParMatrix& A)
{     
      mfem::SparseMatrix mergedA;
      const_cast<mfem::HypreParMatrix*>(&A)->MergeDiagAndOffd(mergedA);
      mfem::Array<int> nonZeroRows;
      for (int i = 0; i < A.NumCols(); i++)
      {
         if (!mergedA.RowIsEmpty(i))
         {
            nonZeroRows.Append(i);
         }
      }
      int numNZRows = nonZeroRows.Size();
      mfem::SparseMatrix nzRowMap(numNZRows, A.GetGlobalNumRows());

      for (int i = 0; i < numNZRows; i++)
      {
         int nzRow_global = nonZeroRows[i] + A.RowPart()[0];
         nzRowMap.Set(i, nzRow_global, 1.0);
      }
      nzRowMap.Finalize();

      auto comm = A.GetComm();
      int rows_part[2];
      int cols_part[2];

      int row_offset;
      MPI_Scan(&numNZRows, &row_offset, 1, MPI_INT, MPI_SUM, comm);

      row_offset -= numNZRows;
      rows_part[0] = row_offset;
      rows_part[1] = row_offset + numNZRows;
      for (int i = 0; i < 2; i++)
      {
         cols_part[i] = A.RowPart()[i];
      }
      int glob_nrows;
      int glob_ncols = A.GetGlobalNumRows();
      MPI_Allreduce(&numNZRows, &glob_nrows, 1, MPI_INT, MPI_SUM, comm);

      return new mfem::HypreParMatrix(comm, numNZRows, glob_nrows, glob_ncols,
                                      nzRowMap.GetI(), nzRowMap.GetJ(), nzRowMap.GetData(), 
                                      rows_part, cols_part); 
      // HypreStealOwnership(*out, nzRowMap);
}

mfem::HypreParMatrix * NonZeroColMap(const mfem::HypreParMatrix& A)
{
      auto At = A.Transpose();
      auto mapT = NonZeroRowMap(*At);
      auto out = mapT->Transpose();
      delete mapT;
      delete At;
      return out;
}
