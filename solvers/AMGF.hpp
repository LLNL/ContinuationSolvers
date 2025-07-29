#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


#ifndef AMGF_HPP
#define AMGF_HPP

namespace mfem
{

/// AMG with filtering
class AMGF : public Solver
{
private:
    MPI_Comm comm;
    int numProcs, myid;
    const HypreParMatrix * A = nullptr;
    const HypreParMatrix * Pc = nullptr;
    const HypreParMatrix * Pnc = nullptr;
    HypreBoomerAMG * amg = nullptr;
    HypreParMatrix * Ac = nullptr;
    Solver * Mfiltered = nullptr;
    bool additive = false;
    int relax_type = 88;
    void Init(MPI_Comm comm_);
    void InitAMG();
    void InitFilteredSpaceSolver();
public:
    AMGF(MPI_Comm comm_);
    AMGF(const Operator & Op, const Operator & P_);
    void SetOperator(const Operator &op);
    void SetContactTransferMap(const Operator & P);
    void SetNonContactTransferMap(const Operator & P);
    void EnableAdditiveCoupling() { additive = true; }
    void EnableMultiplicativeCoupling() { additive = false; }
    void SetAMGRelaxType(int relax_type_) { relax_type = relax_type_;  }

    virtual void Mult(const Vector & y, Vector & x) const; 

    ~AMGF()
    {
        delete amg;
        delete Mfiltered;
        delete Ac;
    }
};

}

#endif // AMGF_HPP
