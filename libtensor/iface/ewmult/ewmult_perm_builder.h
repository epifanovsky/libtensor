#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_EWMULT_PERM_BUILDER_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_EWMULT_PERM_BUILDER_H

#include "../../defs.h"
#include "../../exception.h"
#include "../../core/permutation_builder.h"
#include "../letter_expr.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Builds permutations for the element-wise product

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K>
class ewmult_perm_builder {
public:
    enum {
        NA = N + K,
        NB = M + K,
        NC = N + M + K,
    };

private:
    permutation<NA> m_perma;
    permutation<NB> m_permb;
    permutation<NC> m_permc;

public:
    ewmult_perm_builder(
        const letter_expr<NA> &label_a,
        const permutation<NA> &perm_a,
        const letter_expr<NB> &label_b,
        const permutation<NB> &perm_b,
        const letter_expr<NC> &label_c,
        const letter_expr<K> &ewidx);

    const permutation<NA> &get_perma() const {
        return m_perma;
    }

    const permutation<NB> &get_permb() const {
        return m_permb;
    }

    const permutation<NC> &get_permc() const {
        return m_permc;
    }

};


template<size_t N, size_t M, size_t K>
ewmult_perm_builder<N, M, K>::ewmult_perm_builder(
    const letter_expr<NA> &label_a,
    const permutation<NA> &perm_a,
    const letter_expr<NB> &label_b,
    const permutation<NB> &perm_b,
    const letter_expr<NC> &label_c, const letter_expr<K> &ewidx) {

    sequence<NA, const letter*> seqa1(0), seqa2(0);
    sequence<NB, const letter*> seqb1(0), seqb2(0);
    sequence<NC, const letter*> seqc1(0), seqc2(0);

    size_t k = 0;
    for(size_t i = 0, j = 0; i < NA; i++) {
        const letter &l = label_a.letter_at(i);
        seqa1[i] = &l;
        if(!ewidx.contains(l)) {
            seqc1[k++] = seqa2[j++] = &l;
        }
    }

    for(size_t i = 0, j = 0; i < NB; i++) {
        const letter &l = label_b.letter_at(i);
        seqb1[i] = &l;
        if(!ewidx.contains(l)) {
            seqc1[k++] = seqb2[j++] = &l;
        }
    }
    for(size_t i = 0; i < K; i++) {
        seqc1[N + M + i] = seqa2[N + i] = seqb2[M + i] =
            &ewidx.letter_at(i);
    }

    for(size_t i = 0; i < NC; i++) seqc2[i] = &label_c.letter_at(i);

    m_perma.permute(permutation_builder<NA>(seqa2, seqa1).get_perm());
    m_perma.permute(perm_a);
    m_permb.permute(permutation_builder<NB>(seqb2, seqb1).get_perm());
    m_permb.permute(perm_b);
    m_permc.permute(permutation_builder<NC>(seqc2, seqc1).get_perm());
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_EWMULT_PERM_BUILDER_H
