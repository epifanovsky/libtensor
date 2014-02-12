#ifndef LIBTENSOR_EXPR_EVAL_H
#define LIBTENSOR_EXPR_EVAL_H

#include "eval_selector_i.h"

namespace libtensor {
namespace expr {


/** \defgroup libtensor_expr_eval Basic infrastructure for evaluating
        expressions
    \ingroup libtensor_expr
 **/


/** \brief Generic expression evaluator

    \ingroup libtensor_expr_eval
 **/
class eval {
public:
    /** \brief Evaluates the given expression with a given evaluator selector
     **/
    void evaluate(const expr_tree &e, eval_selector_i &es) const;

    /** \brief Evaluates the given expression with a default evaluator selector
     **/
    void evaluate(const expr_tree &e) const;

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_EVAL_H
