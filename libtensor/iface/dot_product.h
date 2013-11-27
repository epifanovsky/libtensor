#ifndef LIBTENSOR_DOT_PRODUCT_H
#define LIBTENSOR_DOT_PRODUCT_H

#include <libtensor/core/permutation_builder.h>
#include <libtensor/block_tensor/btod_dotprod.h>
#include "labeled_btensor.h"
#include "letter.h"
#include "letter_expr.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Dot product (tensor + tensor)

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A1, bool A2>
double dot_product(
    labeled_btensor<N, T, A1> bt1,
    labeled_btensor<N, T, A2> bt2) {

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
}


/** \brief Dot product (tensor + expression)

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A1>
double dot_product(
    labeled_btensor<N, T, A1> bt1,
    expr<N, T> expr2) {

    anon_eval<N, T> eval2(expr2, bt1.get_label());
    eval2.evaluate();
    return btod_dotprod<N>(bt1.get_btensor(), eval2.get_btensor()).calculate();
}


/** \brief Dot product (expression + tensor)

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A2>
double dot_product(
    expr<N, T> expr1,
    labeled_btensor<N, T, A2> bt2) {

    return dot_product(bt2, expr1);
}


/** \brief Dot product (expression + expression)

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T>
double dot_product(
    expr<N, T> expr1,
    expr<N, T> expr2) {

    std::vector<const letter*> v;
    for(size_t i = 0; i < N; i++) v.push_back(&expr1.letter_at(i));
    letter_expr<N> label(v);
    anon_eval<N, T> eval1(expr1, label);
    eval1.evaluate();
    anon_eval<N, T> eval2(expr2, label);
    eval2.evaluate();
    return btod_dotprod<N>(eval1.get_btensor(), eval2.get_btensor()).
        calculate();
}


} // namespace labeled_btensor_expr

using labeled_btensor_expr::dot_product;

} // namespace libtensor

#endif // LIBTENSOR_DOT_PRODUCT_H
