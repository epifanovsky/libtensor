#ifndef LIBTENSOR_DOT_PRODUCT_H
#define LIBTENSOR_DOT_PRODUCT_H

#include <libtensor/core/permutation_builder.h>
#include <libtensor/block_tensor/btod_dotprod.h>
#include "expr_rhs.h"

namespace libtensor {
namespace iface {


/** \brief Dot product

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
double dot_product(
    const expr_rhs<N, T> &rhs,
    const expr_rhs<N, T> &lhs) {

    /*
    size_t seq1[N], seq2[N];
    for(size_t i = 0; i < N; i++) {
        seq1[i] = i;
        seq2[i] = bt2.index_of(bt1.letter_at(i));
    }
    permutation<N> perma;
    permutation_builder<N> permb(seq1, seq2);
    return btod_dotprod<N>(
        bt1.get_btensor(), perma,
        bt2.get_btensor(), permb.get_perm()).calculate();
        */
    return 0.0;
}


} // namespace iface

using iface::dot_product;

} // namespace libtensor

#endif // LIBTENSOR_DOT_PRODUCT_H
