#ifndef LIBTENSOR_BISPACE_EXPR_OPERATOR_AND_H
#define LIBTENSOR_BISPACE_EXPR_OPERATOR_AND_H

#include "expr.h"
#include "ident.h"
#include "sym.h"

namespace libtensor {
namespace bispace_expr {

template<size_t N, size_t K, size_t NK, typename C>
expr< N * (K + 1), sym< N, K + 1, expr<N, C> > >
inline operator&(
    expr< NK, sym< N, K, expr<N, C> > > lhs,
    expr<N, C> rhs) {

    typedef expr<N, C> core_t;
    typedef sym<N, 1, core_t> sym2_t;
    typedef expr<N, sym2_t> expr2_t;
    typedef sym<N, K + 1, core_t> sym3_t;
    typedef expr<N * (K + 1), sym3_t> expr_t;
    return expr_t(sym3_t(lhs, expr2_t(sym2_t(rhs))));
}


template<size_t N, typename C>
expr< 2 * N, sym< N, 2, expr<N, C> > >
inline operator&(
    expr<N, C> lhs,
    expr<N, C> rhs) {

    typedef expr<N, C> core_t;
    typedef sym<N, 1, core_t> sym_t;
    typedef expr<N, sym_t> expr_t;
    return expr_t(sym_t(core_t(lhs))) & rhs;
}


template<size_t N, size_t K, size_t NK>
expr< N * (K + 1), sym< N, K + 1, expr< N, ident<N> > > >
inline operator&(
    expr< NK, sym< N, K, expr< N, ident<N> > > > lhs,
    const bispace<N> &rhs) {

    typedef expr< N, ident<N> > core_t;
    return lhs & core_t(rhs);
}


template<size_t N>
expr< 2 * N, sym< N, 2, expr< N, ident<N> > > >
inline operator&(
    const bispace<N> &lhs,
    const bispace<N> &rhs) {

    typedef expr< N, ident<N> > core_t;
    typedef sym<N, 1, core_t> sym_t;
    typedef expr<N, sym_t> expr_t;
    return expr_t(sym_t(core_t(lhs))) & rhs;
}


} // namespace bispace_expr

using bispace_expr::operator&;

} // namespace libtensor

#endif // LIBTENSOR_BISPACE_EXPR_OPERATOR_AND_H

