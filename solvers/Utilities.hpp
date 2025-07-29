#include "mfem.hpp"
#ifdef MFEM_USE_STRUMPACK
#include <StrumpackOptions.hpp>
#include <mfem/linalg/strumpack.hpp>
#endif

#ifndef UTILITIES_HPP
#define UTILITIES_HPP


class DirectSolver : public mfem::Solver
{
private:
   mfem::Solver* solver;
#ifdef MFEM_USE_STRUMPACK
    mfem::STRUMPACKRowLocMatrix* Astrumpack;
#endif

public:
   DirectSolver();
   DirectSolver(const mfem::Operator& op);
   virtual ~DirectSolver();

   void SetOperator(const mfem::Operator& op) override;
   void Mult(const mfem::Vector &b, mfem::Vector &x) const override;  
};

mfem::HypreParMatrix * NonZeroRowMap(const mfem::HypreParMatrix& A);
mfem::HypreParMatrix * NonZeroColMap(const mfem::HypreParMatrix& A);

#endif // UTILITIES_HPP