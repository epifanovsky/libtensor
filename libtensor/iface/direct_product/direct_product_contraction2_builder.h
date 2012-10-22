#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIRECT_PRODUCT_CONTRACTION2_BUILDER_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIRECT_PRODUCT_CONTRACTION2_BUILDER_H

#include <libtensor/defs.h>
#include <libtensor/exception.h>
#include <libtensor/core/contraction2.h>
#include <libtensor/core/permutation_builder.h>
#include "../letter_expr.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Builds the contraction2 object using labels
    \tparam N Order of the first %tensor (A).
    \tparam M Order of the second %tensor (B).

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M>
class direct_product_contraction2_builder {
public:
    static const size_t k_ordera = N; //!< Order of the first %tensor
    static const size_t k_orderb = M; //!< Order of the second %tensor
    static const size_t k_orderc = N + M; //!< Order of the result

private:
    contraction2<N, M, 0> m_contr;

public:
    direct_product_contraction2_builder(
        const letter_expr<k_ordera> &label_a,
        const permutation<k_ordera> &perm_a,
        const letter_expr<k_orderb> &label_b,
        const permutation<k_orderb> &perm_b,
        const letter_expr<k_orderc> &label_c);

    const contraction2<N, M, 0> &get_contr() const { return m_contr; }

private:
    static contraction2<N, M, 0> mk_contr(
        const letter_expr<k_ordera> &label_a,
        const letter_expr<k_orderb> &label_b,
        const letter_expr<k_orderc> &label_c);
};


template<size_t N, size_t M>
direct_product_contraction2_builder<N, M>::direct_product_contraction2_builder(
    const letter_expr<k_ordera> &label_a,
    const permutation<k_ordera> &perm_a,
    const letter_expr<k_orderb> &label_b,
    const permutation<k_orderb> &perm_b,
    const letter_expr<k_orderc> &label_c) :

    m_contr(mk_contr(label_a, label_b, label_c)) {

    m_contr.permute_a(perm_a);
    m_contr.permute_b(perm_b);

}


template<size_t N, size_t M>
contraction2<N, M, 0> direct_product_contraction2_builder<N, M>::mk_contr(
    const letter_expr<k_ordera> &label_a,
    const letter_expr<k_orderb> &label_b,
    const letter_expr<k_orderc> &label_c) {

    size_t seq1[k_orderc], seq2[k_orderc];

    for(size_t i = 0; i < k_orderc; i++) seq1[i] = i;

    size_t j = 0, k = 0;
    for(size_t i = 0; i < k_ordera; i++) {
        const letter &l = label_a.letter_at(i);
        if(!label_c.contains(l)) {
            throw_exc("direct_product_contraction2_builder<N, M, K>", "mk_contr()",
                "Inconsistent expression.");
        }
        seq2[j++] = label_c.index_of(l);
    }
    for(size_t i = 0; i < k_orderb; i++) {
        const letter &l = label_b.letter_at(i);
        if(!label_c.contains(l)) {
            throw_exc("direct_product_contraction2_builder<N, M, K>", "mk_contr()",
                "Inconsistent expression.");
        }
        seq2[j++] = label_c.index_of(l);
    }

    permutation_builder<k_orderc> permc(seq1, seq2);
    contraction2<N, M, 0> c(permc.get_perm());
    return c;
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIRECT_PRODUCT_CONTRACTION2_BUILDER_H
