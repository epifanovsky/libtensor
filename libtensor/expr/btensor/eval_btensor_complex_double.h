#ifndef LIBTENSOR_EXPR_EVAL_BTENSOR_COMPLEX_DOUBLE_H
#define LIBTENSOR_EXPR_EVAL_BTENSOR_COMPLEX_DOUBLE_H

#include <complex>
#include <libtensor/expr/dag/expr_tree.h>
#include <libtensor/expr/eval/eval_i.h>

namespace libtensor {
namespace expr {


/** \brief Processor of evaluation plan for btensor result type
        (complex double)

    \ingroup libtensor_expr_btensor
 **/
template<>
class eval_btensor< std::complex<double> > : public eval_i {
public:
    enum {
        Nmax = 8
    };

public:
    /** \brief Virtual destructor
     **/
    virtual ~eval_btensor< std::complex<double> >();

    /** \brief Checks if this evaluator can handle the given expression
     **/
    virtual bool can_evaluate(const expr_tree &e) const;

    /** \brief Evaluates an expression tree
     **/
    virtual void evaluate(const expr_tree &tree) const;

};


} // namespace expr
} // namespace libtensor


#endif // LIBTENSOR_EXPR_EVAL_BTENSOR_COMPLEX_DOUBLE_H
