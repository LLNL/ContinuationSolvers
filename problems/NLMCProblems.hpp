#include "mfem.hpp"
#include "OptProblems.hpp"

#ifndef PROBLEM_DEFS
#define PROBLEM_DEFS

/* Abstract GeneralNLMCProblem class
 * to describe the nonlinear mixed complementarity problem
 * 0 <= x \perp F(x, y) >= 0
 *              Q(x, y)  = 0
 * where NLMC stands for nonlinear mixed complementarity 
 */
class GeneralNLMCProblem
{
protected:
   int dimx, dimy;
   HYPRE_BigInt dimxglb, dimyglb;
   HYPRE_BigInt * dofOffsetsx;
   HYPRE_BigInt * dofOffsetsy;
   int label;
public:
   GeneralNLMCProblem();
   virtual void Init(HYPRE_BigInt * dofOffsetsx_, HYPRE_BigInt * dofOffsetsy_);
   virtual void F(const mfem::Vector &x, const mfem::Vector &y, mfem::Vector &feval, int &eval_err) const = 0;
   virtual void Q(const mfem::Vector &x, const mfem::Vector &y, mfem::Vector &qeval, int &eval_err) const = 0;
   virtual mfem::Operator * DxF(const mfem::Vector &x, const mfem::Vector &y) = 0;
   virtual mfem::Operator * DyF(const mfem::Vector &x, const mfem::Vector &y) = 0;
   virtual mfem::Operator * DxQ(const mfem::Vector &x, const mfem::Vector &y) = 0;
   virtual mfem::Operator * DyQ(const mfem::Vector &x, const mfem::Vector &y) = 0;
   int GetDimx() const { return dimx; };
   int GetDimy() const { return dimy; }; 
   HYPRE_BigInt GetDimxGlb() const { return dimxglb; };
   HYPRE_BigInt GetDimyGlb() const { return dimyglb; };
   HYPRE_BigInt * GetDofOffsetsx() const { return dofOffsetsx; };
   HYPRE_BigInt * GetDofOffsetsy() const { return dofOffsetsy; }; 
   void setProblemLabel(int label_) { label = label_; };
   int getProblemLabel() { return label; };
   virtual ~GeneralNLMCProblem();
};


class OptNLMCProblem : public GeneralNLMCProblem
{
protected:
   OptProblem * optproblem;
   mfem::HypreParMatrix * dFdx;
   mfem::HypreParMatrix * dFdy;
   mfem::HypreParMatrix * dQdx;
   mfem::HypreParMatrix * dQdy;
   mfem::HypreParMatrix * Pc;

public:
   OptNLMCProblem(OptProblem * problem_);
   void F(const mfem::Vector &x, const mfem::Vector &y, mfem::Vector &feval, int &eval_err) const;
   void Q(const mfem::Vector &x, const mfem::Vector &y, mfem::Vector &qeval, int &eval_err) const;
   mfem::Operator * DxF(const mfem::Vector &x, const mfem::Vector &y);
   mfem::Operator * DyF(const mfem::Vector &x, const mfem::Vector &y);
   mfem::Operator * DxQ(const mfem::Vector &x, const mfem::Vector &y);
   mfem::Operator * DyQ(const mfem::Vector &x, const mfem::Vector &y);
   OptProblem * GetOptProblem() { return optproblem;  };
   virtual ~OptNLMCProblem();    
};


class EqualityConstrainedHomotopyProblem : public GeneralNLMCProblem
{
protected:
   mfem::HypreParMatrix * dFdx;
   mfem::HypreParMatrix * dFdy;
   mfem::HypreParMatrix * dQdx;
   mfem::HypreParMatrix * dQdy;
   int dimu;
   int dimc;
   int dimcglb;
   mfem::Array<int> y_partition;
   HYPRE_BigInt * uOffsets = nullptr;
   HYPRE_BigInt * cOffsets = nullptr;
   bool set_sizes = false;
public:
   EqualityConstrainedHomotopyProblem();
   void SetSizes(int dimu_, int dimc_);
   virtual mfem::Vector residual(const mfem::Vector & u) const = 0;
   virtual mfem::Vector jTvp(const mfem::Vector &u, const mfem::Vector & l) const = 0;
   virtual mfem::HypreParMatrix * residualJacobian(const mfem::Vector & u) = 0; 
   virtual mfem::Vector constraint(const mfem::Vector & u) const = 0;
   virtual mfem::HypreParMatrix * constraintJacobian(const mfem::Vector & u) = 0;
   void F(const mfem::Vector &x, const mfem::Vector &y, mfem::Vector &feval, int &eval_err) const;
   void Q(const mfem::Vector &x, const mfem::Vector &y, mfem::Vector &qeval, int &eval_err) const;
   mfem::Operator * DxF(const mfem::Vector &/*x*/, const mfem::Vector &/*y*/) { return dFdx; };
   mfem::Operator * DyF(const mfem::Vector &/*x*/, const mfem::Vector &/*y*/) { return dFdy; };
   mfem::Operator * DxQ(const mfem::Vector &/*x*/, const mfem::Vector &/*y*/) { return dQdx; };
   mfem::Operator * DyQ(const mfem::Vector &x, const mfem::Vector &y);
   virtual ~EqualityConstrainedHomotopyProblem();    
};




#endif
