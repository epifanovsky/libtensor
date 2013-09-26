#ifndef LIBTENSOR_IFACE_LABELED_LHS_RHS_H
#define LIBTENSOR_IFACE_LABELED_LHS_RHS_H

#include "expr/expr_rhs.h"
#include "labeled_lhs.h"

namespace libtensor {
namespace iface {

using labeled_btensor_expr::expr_rhs; // temporary


/** \brief Labeled tensor or expression that can be either on the left-hand side
        or on the right-hand side of a tensor expression

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
class labeled_lhs_rhs :
    virtual public labeled_lhs<N, T>, virtual public expr_rhs<N, T> {

public:
    labeled_lhs_rhs(expr_lhs<N, T> &lhs, const letter_label<N> &label,
        expr_rhs<N, T> &rhs) :
        labeled_lhs(lhs, label), expr_rhs(rhs)
    { }

};


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_LABELED_LHS_RHS_H
