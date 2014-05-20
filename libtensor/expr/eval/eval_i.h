#ifndef LIBTENSOR_EXPR_EVAL_I_H
#define LIBTENSOR_EXPR_EVAL_I_H

#include <libtensor/expr/dag/expr_tree.h>

namespace libtensor {
namespace expr {


/** \brief Evaluator interface

    \ingroup libtensor_expr_eval
 **/
class eval_i {
public:
    /** \brief Virtual destructor
     **/
    virtual ~eval_i() { }

    /** \brief Returns true if this evaluator can handle the given expression
     **/
    virtual bool can_evaluate(const expr::expr_tree &e) const = 0;

    /** \brief Evaluates the given expression
     **/
    virtual void evaluate(const expr::expr_tree &e) const = 0;

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_EVAL_I_H
