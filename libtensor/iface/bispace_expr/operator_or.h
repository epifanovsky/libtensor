#ifndef LIBTENSOR_BISPACE_EXPR_OPERATOR_OR_H
#define LIBTENSOR_BISPACE_EXPR_OPERATOR_OR_H

#include "expr.h"
#include "ident.h"
#include "asym.h"

namespace libtensor {
namespace bispace_expr {


template<size_t N1, size_t N2, typename C1, typename C2>
expr< N1 + N2, asym<N1, N2, C1, C2> >
inline operator|(
    expr<N1, C1> lhs,
    expr<N2, C2> rhs) {

    typedef asym<N1, N2, C1, C2> asym_t;
    typedef expr<N1 + N2, asym_t> expr_t;
    return expr_t(asym_t(lhs, rhs));
}


template<size_t N1, size_t N2, typename C1>
expr< N1 + N2, asym< N1, N2, C1, ident<N2> > >
inline operator|(
    expr<N1, C1> lhs,
    const bispace<N2> &rhs) {

    typedef ident<N2> id2_t;
    typedef expr<N2, id2_t> expr2_t;
    return lhs | expr2_t(id2_t(rhs));
}


template<size_t N1, size_t N2, typename C2>
expr< N1 + N2, asym< N1, N2, ident<N1>, C2 > >
inline operator|(
    const bispace<N1> &lhs,
    expr<N2, C2> rhs) {

    typedef ident<N1> id1_t;
    typedef expr<N1, id1_t> expr1_t;
    return expr1_t(id1_t(lhs)) | rhs;
}


template<size_t N1, size_t N2>
expr< N1 + N2, asym< N1, N2, ident<N1>, ident<N2> > >
inline operator|(
    const bispace<N1> &lhs,
    const bispace<N2> &rhs) {

    typedef ident<N1> id1_t;
    typedef expr<N1, id1_t> expr1_t;
    typedef ident<N2> id2_t;
    typedef expr<N2, id2_t> expr2_t;
    return expr1_t(id1_t(lhs)) | expr2_t(id2_t(rhs));
}


} // namespace bispace_expr

using bispace_expr::operator|;

} // namespace libtensor

#endif // LIBTENSOR_BISPACE_EXPR_OPERATOR_OR_H
