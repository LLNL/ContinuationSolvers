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
   virtual mfem::HypreParMatrix * DxF(const mfem::Vector &x, const mfem::Vector &y) = 0;
   virtual mfem::HypreParMatrix * DyF(const mfem::Vector &x, const mfem::Vector &y) = 0;
   virtual mfem::HypreParMatrix * DxQ(const mfem::Vector &x, const mfem::Vector &y) = 0;
   virtual mfem::HypreParMatrix * DyQ(const mfem::Vector &x, const mfem::Vector &y) = 0;
   int GetDimx() const { return dimx; };
   int GetDimy() const { return dimy; }; 
   HYPRE_BigInt GetDimxGlb() const { return dimxglb; };
   HYPRE_BigInt GetDimyGlb() const { return dimyglb; };
   HYPRE_BigInt * GetDofOffsetsx() const { return dofOffsetsx; };
   HYPRE_BigInt * GetDofOffsetsy() const { return dofOffsetsy; }; 
   void setProblemLabel(int label_) { label = label_; };
   int getProblemLabel() { return label; };
   virtual mfem::HypreParMatrix * GetRestrictionToConstrainedDofs() = 0;
   ~GeneralNLMCProblem();
};


class OptNLMCProblem : public GeneralNLMCProblem
{
protected:
   ParOptProblem * optproblem;
   mfem::HypreParMatrix * dFdx;
   mfem::HypreParMatrix * dFdy;
   mfem::HypreParMatrix * dQdx;
   mfem::HypreParMatrix * dQdy;
   mfem::HypreParMatrix * Pc;

public:
   OptNLMCProblem(ParOptProblem * problem_);
   void F(const mfem::Vector &x, const mfem::Vector &y, mfem::Vector &feval, int &eval_err) const;
   void Q(const mfem::Vector &x, const mfem::Vector &y, mfem::Vector &qeval, int &eval_err) const;
   mfem::HypreParMatrix * DxF(const mfem::Vector &x, const mfem::Vector &y);
   mfem::HypreParMatrix * DyF(const mfem::Vector &x, const mfem::Vector &y);
   mfem::HypreParMatrix * DxQ(const mfem::Vector &x, const mfem::Vector &y);
   mfem::HypreParMatrix * DyQ(const mfem::Vector &x, const mfem::Vector &y);
   ParOptProblem * GetOptProblem() { return optproblem;  };
   mfem::HypreParMatrix * GetRestrictionToConstrainedDofs() override;
   ~OptNLMCProblem();    
};



#endif
