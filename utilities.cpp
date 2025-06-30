#include "mfem.hpp"
#include "utilities.hpp"


mfem::HypreParMatrix * GenerateHypreParMatrixFromSparseMatrix(HYPRE_BigInt * colOffsetsloc, HYPRE_BigInt * rowOffsetsloc, mfem::SparseMatrix * Asparse)
{
  int ncols_loc = colOffsetsloc[1] - colOffsetsloc[0];
  int nrows_loc = rowOffsetsloc[1] - rowOffsetsloc[0];
  HYPRE_BigInt ncols_glb, nrows_glb;
  MPI_Allreduce(&nrows_loc, &nrows_glb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&ncols_loc, &ncols_glb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  int * AI          = Asparse->GetI();
  HYPRE_BigInt * AJ = Asparse->GetJ();
  double * Adata    = Asparse->GetData();

  mfem::HypreParMatrix * Ahypre = nullptr;
  Ahypre = new mfem::HypreParMatrix(MPI_COMM_WORLD, nrows_loc, nrows_glb, ncols_glb, AI, AJ, Adata, rowOffsetsloc, colOffsetsloc);
  return Ahypre;
}


mfem::HypreParMatrix * GenerateHypreParMatrixFromDiagonal(HYPRE_BigInt * offsetsloc, 
		mfem::Vector & diag)
{
   int n_loc = offsetsloc[1] - offsetsloc[0];
   int n_glb = 0;
   MPI_Allreduce(&n_loc, &n_glb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   
   mfem::SparseMatrix * Dsparse = new mfem::SparseMatrix(n_loc, n_glb);
   mfem::Array<int> cols;
   mfem::Vector entries;
   cols.SetSize(1);
   entries.SetSize(1);
   for(int j = 0; j < n_loc; j++)
   {
     cols[0] = offsetsloc[0] + j;
     entries(0) = diag(j);
     Dsparse->SetRow(j, cols, entries);
   }   
   Dsparse->Finalize();
   mfem::HypreParMatrix * Dhypre = nullptr;
   Dhypre = GenerateHypreParMatrixFromSparseMatrix(offsetsloc, offsetsloc, Dsparse);
   delete Dsparse;
   return Dhypre;   
}

mfem::HypreParMatrix * GenerateProjector(HYPRE_BigInt * offsets, HYPRE_BigInt * reduced_offsets, HYPRE_Int * mask)
{
  int n_cols_loc = offsets[1] - offsets[0];
  int n_cols_glb = 0;
  MPI_Allreduce(&n_cols_loc, &n_cols_glb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  int n_rows_loc = reduced_offsets[1] - reduced_offsets[0];

  mfem::SparseMatrix * Psparse = new mfem::SparseMatrix(n_rows_loc, n_cols_glb);
  mfem::Array<int> cols;
  mfem::Vector entries;
  cols.SetSize(1);
  entries.SetSize(1);

  int row = 0;
  for(int j = 0; j < n_cols_loc; j++)
  {
    if (mask[j] == 1)
    {
      cols[0] = offsets[0] + j;
      entries(0) = 1.0;
      Psparse->SetRow(row, cols, entries);
      row += 1;
    }
  }
  Psparse->Finalize();
  mfem::HypreParMatrix * Phypre = nullptr;
  Phypre = GenerateHypreParMatrixFromSparseMatrix(offsets, reduced_offsets, Psparse);
  delete Psparse;
  return Phypre;
}

mfem::HypreParMatrix * GenerateProjector(HYPRE_BigInt * offsets, HYPRE_BigInt * reduced_offsets, const mfem::HypreParVector & mask)
{
  int n_cols_loc = offsets[1] - offsets[0];
  int n_cols_glb = 0;
  MPI_Allreduce(&n_cols_loc, &n_cols_glb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  int n_rows_loc = reduced_offsets[1] - reduced_offsets[0];

  mfem::SparseMatrix * Psparse = new mfem::SparseMatrix(n_rows_loc, n_cols_glb);
  mfem::Array<int> cols;
  mfem::Vector entries;
  cols.SetSize(1);
  entries.SetSize(1);

  int row = 0;
  for(int j = 0; j < n_cols_loc; j++)
  {
    if (mask(j) > 0.5)
    {
      cols[0] = offsets[0] + j;
      entries(0) = 1.0;
      Psparse->SetRow(row, cols, entries);
      row += 1;
    }
  }
  Psparse->Finalize();
  mfem::HypreParMatrix * Phypre = nullptr;
  Phypre = GenerateHypreParMatrixFromSparseMatrix(offsets, reduced_offsets, Psparse);
  delete Psparse;
  return Phypre;
}




HYPRE_BigInt * offsetsFromLocalSizes(int n, MPI_Comm comm)
{
  
  int nprocs = 0;
  int myrank = 0;
  MPI_Comm_rank(comm, &myrank);
  MPI_Comm_size(comm, &nprocs);
  
  HYPRE_BigInt * offsets = new HYPRE_BigInt[2];
  if (myrank == 0)
  {
    offsets[0] = 0;
    offsets[1] = n;
  }
  else
  {
    offsets[0] = 0;
    offsets[1] = 0;
  }
  
  // receive then send
  
  // Receive local size info from processes with rank less than myrank 
  // Populate that as entries of helper
  HYPRE_BigInt * helper;
  if (myrank > 0)
  {
    helper = new HYPRE_BigInt[static_cast<size_t>(myrank)];
  }
  int tag;
  for (int i = 0; i < myrank; i++)
  {
    tag = myrank + i * nprocs;
    MPI_Recv (&(helper[i]), 1, MPI_INT, i, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    offsets[0] += helper[i];
  }

  if (myrank > 0)
  {
    delete[] helper;
  }
  offsets[1] = offsets[0] + n;
  
  // Send local size info to all processes with rank greater than myrank
  for (int i = myrank + 1; i < nprocs; i++)
  {
    tag = i + myrank * nprocs;
    MPI_Send (&n, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
  }
  return offsets;
}


void HypreToMfemOffsets(HYPRE_BigInt * offsets)
{
  if (offsets[1] < offsets[0])
  {
    offsets[1] = offsets[0];
  }
  else
  {
    offsets[1] = offsets[1] + 1;
  }
}

