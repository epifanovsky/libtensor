#ifndef LIBTENSOR_EXPR_LABELED_LHS_RHS_H
#define LIBTENSOR_EXPR_LABELED_LHS_RHS_H

#include "expr_rhs.h"
#include "expr_lhs.h"

namespace libtensor {
namespace expr {


/** \brief Labeled tensor or expression that can be either on the left-hand side
        or on the right-hand side of a tensor expression

    \ingroup libtensor_expr_iface
 **/
template<size_t N, typename T>
class labeled_lhs_rhs :
    virtual public labeled_lhs<N, T>, virtual public expr_rhs<N, T> {

public:
    labeled_lhs_rhs(expr_lhs<N, T> &lhs, const label<N> &l,
        const expr_rhs<N, T> &rhs) :
        labeled_lhs<N, T>(lhs, l), expr_rhs<N, T>(rhs)
    { }

    const expr_rhs<N, T> &operator=(const expr_rhs<N, T> &rhs) {
        return labeled_lhs<N, T>::operator=(rhs);
    }

    const expr_rhs<N, T> &operator=(const labeled_lhs_rhs<N, T> &rhs) {
        return labeled_lhs<N, T>::operator=((const expr_rhs<N, T>&)rhs);
    }

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_LABELED_LHS_RHS_H
