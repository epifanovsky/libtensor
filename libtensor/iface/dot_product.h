#ifndef LIBTENSOR_DOT_PRODUCT_H
#define LIBTENSOR_DOT_PRODUCT_H

#include "../defs.h"
#include "../exception.h"
#include "../core/permutation_builder.h"
#include <libtensor/block_tensor/btod/btod_dotprod.h>
#include "labeled_btensor.h"
#include "letter.h"
#include "letter_expr.h"

namespace libtensor {

namespace labeled_btensor_expr {


/** \brief Dot product (%tensor + %tensor)

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool Assignable1, bool Assignable2>
double dot_product(
    labeled_btensor<N, T, Assignable1> bt1,
    labeled_btensor<N, T, Assignable2> bt2) {

    size_t seq1[N], seq2[N];
    for(size_t i = 0; i < N; i++) {
        seq1[i] = i;
        seq2[i] = bt2.index_of(bt1.letter_at(i));
    }
    permutation<N> perma;
    permutation_builder<N> permb(seq1, seq2);
    btod_dotprod<N> op(
        bt1.get_btensor(), perma, bt2.get_btensor(), permb.get_perm());
    return op.calculate();
}


/** \brief Dot product (%tensor + expression)

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool Assignable1, typename Expr2>
double dot_product(
    labeled_btensor<N, T, Assignable1> bt1,
    expr<N, T, Expr2> expr2) {

    anon_eval<N, T, Expr2> eval2(expr2, bt1.get_label());
    eval2.evaluate();
    return btod_dotprod<N>(bt1.get_btensor(), eval2.get_btensor()).calculate();
}


/** \brief Dot product (expression + %tensor)

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename Expr1, bool Assignable2>
double dot_product(
    expr<N, T, Expr1> expr1,
    labeled_btensor<N, T, Assignable2> bt2) {

    return dot_product(bt2, expr1);
}


/** \brief Dot product (expression + expression)

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename Expr1, typename Expr2>
double dot_product(
    expr<N, T, Expr1> expr1,
    expr<N, T, Expr2> expr2) {

    return 0.0;
}


} // namespace labeled_btensor_expr

using labeled_btensor_expr::dot_product;

} // namespace libtensor

#endif // LIBTENSOR_DOT_PRODUCT_H
