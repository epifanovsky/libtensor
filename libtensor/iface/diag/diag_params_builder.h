#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_PARAMS_BUILDER_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_PARAMS_BUILDER_H

#include <libtensor/core/permutation_builder.h>
#include "../letter_expr.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Builds the parameters for btod_diag using labels
    \tparam N Order of the RHS %tensor.
    \tparam M Order of the diagonal.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M>
class diag_params_builder {
public:
    static const size_t k_ordera = N; //!< Order of the LHS %tensor
    static const size_t k_orderb = N - M + 1; //!< Order of the RHS %tensor

private:
    mask<N> m_msk;
    permutation<k_orderb> m_perm;

public:
    diag_params_builder(
        const letter_expr<k_ordera> &label_a,
        const permutation<k_ordera> &perm_a,
        const letter_expr<k_orderb> &label_b,
        const letter &letter_diag,
        const letter_expr<M> &label_diag);

    const mask<N> &get_mask() const { return m_msk; }
    const permutation<k_orderb> &get_perm() const { return m_perm; }

};


template<size_t N, size_t M>
diag_params_builder<N, M>::diag_params_builder(
    const letter_expr<k_ordera> &label_a,
    const permutation<k_ordera> &perm_a,
    const letter_expr<k_orderb> &label_b,
    const letter &letter_diag, const letter_expr<M> &label_diag) {

    sequence<k_ordera, size_t> mapa(0);
    for(register size_t i = 0; i < k_ordera; i++) mapa[i] = i;
    perm_a.apply(mapa);

    for(size_t i = 0; i < M; i++)
        m_msk[mapa[label_a.index_of(label_diag.letter_at(i))]] = true;

    sequence<k_orderb, size_t> seq1(0), seq2(0);
    bool first = true;
    size_t j = 0;
    for(size_t i = 0; i < k_ordera; i++) {
        const letter &l = label_a.letter_at(mapa[i]);
        if(label_diag.contains(l)) {
            if(first) {
                seq1[j] = j;
                seq2[j] = label_b.index_of(letter_diag);
                j++;
                first = false;
            }
        } else {
            seq1[j] = j;
            seq2[j] = label_b.index_of(l);
            j++;
        }
    }
    permutation_builder<k_orderb> pb(seq1, seq2);
    m_perm.permute(pb.get_perm());
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_PARAMS_BUILDER_H
