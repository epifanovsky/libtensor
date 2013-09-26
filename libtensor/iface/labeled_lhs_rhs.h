#ifndef LIBTENSOR_IFACE_LABELED_LHS_RHS_H
#define LIBTENSOR_IFACE_LABELED_LHS_RHS_H

#include "expr_rhs.h"
#include "expr_lhs.h"

namespace libtensor {
namespace iface {


/** \brief Labeled tensor or expression that can be either on the left-hand side
        or on the right-hand side of a tensor expression

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
class labeled_lhs_rhs :
    virtual public labeled_lhs<N, T>, virtual public expr_rhs<N, T> {

public:
    labeled_lhs_rhs(expr_lhs<N, T> &lhs, const letter_expr<N> &label,
        const expr_rhs<N, T> &rhs) :
        labeled_lhs<N, T>(lhs, label), expr_rhs<N, T>(rhs)
    { }

    using labeled_lhs<N, T>::operator=;

};


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_LABELED_LHS_RHS_H
