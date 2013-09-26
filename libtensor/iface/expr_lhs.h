#ifndef LIBTENSOR_IFACE_EXPR_LHS_H
#define LIBTENSOR_IFACE_EXPR_LHS_H

#include "expr/expr_rhs.h"

namespace libtensor {
namespace iface {

using labeled_btensor_expr::expr_rhs; // temporary


/** \brief Assignable left-hand-side of a tensor expression

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
class expr_lhs {
public:
    /** \brief Performs the assignment of the right-hand-side to this
            left-hand-side of the expression
     **/
    virtual void assign(expr_rhs<N, T> &rhs, const letter_expr<N> &label) = 0;

};


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_EXPR_LHS_H
