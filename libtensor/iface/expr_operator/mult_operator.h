#ifndef LIBTENSOR_IFACE_MULT_OPERATOR_H
#define LIBTENSOR_IFACE_MULT_OPERATOR_H

#include "../expr_core/mult_core.h"

namespace libtensor {
namespace iface {


/** \brief Element-wise multiplication of two expressions

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
expr_rhs<N, T> mult(
    expr_rhs<N, T> lhs,
    expr_rhs<N, T> rhs) {

    return expr_rhs<N, T>(new mult_core<N, T>(lhs, rhs, false));
}


/** \brief Element-wise division of two expressions

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
expr_rhs<N, T> div(
    expr_rhs<N, T> lhs,
    expr_rhs<N, T> rhs) {

    return expr_rhs<N, T>(new mult_core<N, T>(lhs, rhs, true));
}


/** \brief Element-wise multiplication of two expressions

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, size_t K, typename T>
expr_rhs<N + M - K, T> ewmult(
    const letter_expr<K> &ewidx,
    const expr_rhs<N, T> &bta,
    const expr_rhs<M, T> &btb) {

    typedef ewmult_core<N - K, M - K, K, T> ewmult_core_t;

    static const char *method = "ewmult(const letter_expr<K> &, "
            "const expr_rhs<N, T> &, const expr_rhs<M, T> &)";

    sequence<2 * K, size_t> ewseq(0);
    std::vector<const letter *> label;
    for (size_t i = 0; i < K; i++) {
        const letter &l = ewidx.letter_at(i);
        if (! bta.contains(l) || ! btb.contains(l)) {
            throw expr_exception(g_ns, "", method, __FILE__, __LINE__,
                    "Letter not found.");
        }

        ewseq[i] = bta.index_of(l);
        ewseq[i + K] = btb.index_of(l);
    }

    return expr_rhs<N + M - K, T>(
            new ewmult_core_t(ewseq, bta.get_core(), btb.get_core()),
            letter_expr<N + M - K>(label));
}


/** \brief Element-wise multiplication of two expressions

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N + M - 1, T> ewmult(
    const letter &l,
    const expr_rhs<N, T> &bta,
    const expr_rhs<M, T> &btb) {

    return ewmult(letter_expr<1>(l), bta, btb);
}

} // namespace iface

using iface::mult;

} // namespace libtensor

#endif // LIBTENSOR_IFACE_MULT_OPERATOR_H
