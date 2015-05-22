#ifndef LIBTENSOR_EXPR_OPERATORS_DIAG_H
#define LIBTENSOR_EXPR_OPERATORS_DIAG_H

#include <libtensor/expr/dag/node_diag.h>
#include <libtensor/expr/expr_exception.h>

namespace libtensor {
namespace expr {


/** \brief Extraction of a general tensor diagonal
    \tparam N Tensor order.
    \tparam K Partial order of diagonal in result
    \tparam M Total diagonal order
    \tparam T Tensor element type.

    Extracts a general tensor diagonal according to the first two labels
    provided. The first label defines the diagonal indexes which will
    end up in the result, while the second specifies over which tensor
    dimensions the diagonal(s) will be taken. Thus, the second label has to
    contain all letters which are present in the first label. These letters
    are arranged in the second label such that each letter is followed by
    all other letters over which the respective diagonal should be taken.
    This is best illustrated by examples:
    -# Extract a single diagonal of a matrix \f$ d_i = m_{ii} \f$
    \code
    letter i, j;
    d(i) = diag(i, i|j, m(i|j));
    \endcode
    -# Extract the 2D diagonal of a 4D tensor \f$ d_{ia} = m_{iaia} \f$
    \code
    letter i, j, a, b;
    d(i|a) = diag(i|a, i|j|a|b, m(i|a|j|b));
    \endcode
    -# Extract two general diagonals of a 6D tensor \f$ d_{pai} = m_{iaapia}
    \f[
    letter i, j, a, b, c, p;
    d(i|a|p) = diag(i|a, i|j|a|c|b, m(i|b|a|p|j|c));
    \f]

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, size_t K, typename T>
expr_rhs<N - M + K, T> diag(
    const label<K> &l1,
    const label<M> &l2,
    const expr_rhs<N, T> &subexpr) {

    static const char method[] = "diag(const label<ND> &, "
            "const label<NTD> &, const expr_rhs<NA, T> &)";

    enum {
        NA = N,
        ND = K,
        NTD = M,
        NC = N - M + K
    };

    // Translate l1 and l2 in vector of diagonals [0011122...] and diagonal index
    std::vector<size_t> d(M, 0), didx(K, 0);
    if (! l1.contains(l2.letter_at(0))) {
        throw expr_exception(g_ns, "", method, __FILE__, __LINE__,
                "First letter in l2 not found in l1.");
    }
    size_t j = 0;
    for(size_t i = 1, nd = 1; i < M; i++, nd++) {
        const letter &l = l2.letter_at(i);
        if(l1.contains(l)) {
            if (nd == 1) {
                throw expr_exception(g_ns, "", method, __FILE__, __LINE__,
                        "Diagonal of order 1.");
            }
            nd = 0;
            j++;
            didx[j] = l1.index_of(l);
        }
        d[i] = j;
    }
    if (j + 1 != ND) {
        throw expr_exception(g_ns, "", method, __FILE__, __LINE__,
                "Number of diagonals.");
    }

    j = 0;

    // Construct tensor index, output index, and diagonal index
    std::vector<size_t> idx(N, 0), oidx(NC, 0);
    for(size_t i = 0; i < N; i++) {
        const letter &l = subexpr.letter_at(i);
        if(! l2.contains(l)) { idx[i] = i + K; }
        else { idx[i] = d[l2.index_of(l)]; j++; }
    }
    if (j != NTD) {
        throw expr_exception(g_ns, "", method, __FILE__, __LINE__,
                "Unknown index in l2.");
    }

    node_diag ndiag(NC, idx, didx);
    ndiag.build_output_indices(oidx);

    std::vector<const letter*> lab(NC, 0);
    for(size_t i = 0; i < NC; i++) {
        if(oidx[i] < K) lab[i] = &l1.letter_at(didx[oidx[i]]);
        else lab[i] = &subexpr.letter_at(oidx[i] - 1);
    }

    expr_tree e(ndiag);
    e.add(e.get_root(), subexpr.get_expr());
    return expr_rhs<NC, T>(e, label<NC>(lab));
}


/** \brief Extraction of a general tensor diagonal
    \tparam N Tensor order.
    \tparam M Diagonal order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N - M + 1, T> diag(
    const letter &l1,
    const label<M> &l2,
    const expr_rhs<N, T> &subexpr) {

    return diag(label<1>(l1), l2, subexpr);
}


} // namespace expr

} // namespace libtensor


namespace libtensor {

using expr::diag;

} // namespace libtensor

#endif // LIBTENSOR_EXPR_OPERATORS_DIAG_H
