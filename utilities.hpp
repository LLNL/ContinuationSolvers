#include "mfem.hpp"

#ifndef UTILITY_FUNCTIONS
#define UTILITY_FUNCTIONS

void HypreToMfemOffsets(HYPRE_BigInt * offsets);

mfem::HypreParMatrix * GenerateHypreParMatrixFromSparseMatrix(HYPRE_BigInt * colOffsetsloc, HYPRE_BigInt * rowOffsetsloc, mfem::SparseMatrix * Asparse);

mfem::HypreParMatrix * GenerateHypreParMatrixFromDiagonal(HYPRE_BigInt * offsetsloc, 
		mfem::Vector & diag);


mfem::HypreParMatrix * GenerateProjector(HYPRE_BigInt * offsets, HYPRE_BigInt * reduced_offsets, HYPRE_Int * mask);

mfem::HypreParMatrix * GenerateProjector(HYPRE_BigInt * offsets, HYPRE_BigInt * reduced_offsets, const mfem::HypreParVector & mask);


HYPRE_BigInt * offsetsFromLocalSizes(int n, MPI_Comm comm = MPI_COMM_WORLD);


#endif
