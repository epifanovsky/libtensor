#ifndef LIBTENSOR_EXPR_EVAL_BTENSOR_H
#define LIBTENSOR_EXPR_EVAL_BTENSOR_H

#include <libtensor/expr/dag/expr_tree.h>
#include <libtensor/expr/eval/eval_i.h>

namespace libtensor {
namespace expr {


/** \brief Processor of evaluation plan for btensor result type
    \tparam T Tensor element type.

    \ingroup libtensor_expr_btensor
 **/
//template<typename T> class eval_btensor;
//template class eval_btensor<T>;

template<typename T>
class eval_btensor : public eval_i {
public:
    enum {
        Nmax = 8
    };

public:
    /** \brief Virtual destructor
     **/
    virtual ~eval_btensor();

    /** \brief Checks if this evaluator can handle the given expression
     **/
    virtual bool can_evaluate(const expr_tree &e) const;

    /** \brief Evaluates an expression tree
     **/
    virtual void evaluate(const expr_tree &tree) const;

public:
    /** \brief Specifies whether to use libxm contractions (if available)
     **/
    static void use_libxm(bool usexm);

};


} // namespace expr
} // namespace libtensor

//#include "eval_btensor_double.h"
//#include "eval_btensor_float.h"

#endif // LIBTENSOR_EXPR_EVAL_BTENSOR_H
