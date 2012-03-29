#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM_OPERATOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM_OPERATOR_H

#include "symm1_core.h"
#include "symm1_eval.h"
#include "symm2_core.h"
#include "symm2_eval.h"
#include "symm3_core.h"
#include "symm3_eval.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Symmetrization of an expression over two sets of indexes
    \tparam N Tensor order.
    \tparam M Number of indexes in each set.
    \tparam T Tensor element type.
    \tparam SubCore Sub-expression core.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, typename SubCore>
inline
expr< N, T, symm2_core<N, M, true, T, SubCore> >
symm(
    const letter_expr<M> sym1,
    const letter_expr<M> sym2,
    expr<N, T, SubCore> subexpr) {

    typedef symm2_core<N, M, true, T, SubCore> core_t;
    typedef expr<N, T, core_t> expr_t;
    return expr_t(core_t(sym1, sym2, subexpr));
}


/** \brief Anti-symmetrization of an expression over two sets of indexes
    \tparam N Tensor order.
    \tparam M Number of indexes in each set.
    \tparam T Tensor element type.
    \tparam SubCore Sub-expression core.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, typename SubCore>
inline
expr< N, T, symm2_core<N, M, false, T, SubCore> >
asymm(
    const letter_expr<M> sym1,
    const letter_expr<M> sym2,
    expr<N, T, SubCore> subexpr) {

    typedef symm2_core<N, M, false, T, SubCore> core_t;
    typedef expr<N, T, core_t> expr_t;
    return expr_t(core_t(sym1, sym2, subexpr));
}


/** \brief Symmetrization of an expression over three indexes
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam SubCore Sub-expression core.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename SubCore>
inline
expr< N, T, symm3_core<N, true, T, SubCore> >
symm(
    const letter &l1,
    const letter &l2,
    const letter &l3,
    expr<N, T, SubCore> subexpr) {

    typedef symm3_core<N, true, T, SubCore> core_t;
    typedef expr<N, T, core_t> expr_t;
    return expr_t(core_t(l1, l2, l3, subexpr));
}


/** \brief Anti-symmetrization of an expression over three indexes
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam SubCore Sub-expression core.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename SubCore>
inline
expr< N, T, symm3_core<N, false, T, SubCore> >
asymm(
    const letter &l1,
    const letter &l2,
    const letter &l3,
    expr<N, T, SubCore> subexpr) {

    typedef symm3_core<N, false, T, SubCore> core_t;
    typedef expr<N, T, core_t> expr_t;
    return expr_t(core_t(l1, l2, l3, subexpr));
}


/** \brief Symmetrization of an expression over two indexes
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam SubCore Sub-expression core.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename SubCore>
inline
expr< N, T, symm2_core<N, 1, true, T, SubCore> >
symm(
    const letter &l1,
    const letter &l2,
    expr<N, T, SubCore> subexpr) {

    return symm(letter_expr<1>(l1), letter_expr<1>(l2), subexpr);
}


/** \brief Anti-symmetrization of an expression over two indexes
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam SubCore Sub-expression core.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename SubCore>
inline
expr< N, T, symm2_core<N, 1, false, T, SubCore> >
asymm(
    const letter &l1,
    const letter &l2,
    expr<N, T, SubCore> subexpr) {

    return asymm(letter_expr<1>(l1), letter_expr<1>(l2), subexpr);
}


/** \brief Symmetrization of a %tensor over two sets of indexes
    \tparam N Tensor order.
    \tparam M Number of indexes in each set.
    \tparam T Tensor element type.
    \tparam A Tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, bool A>
inline
expr< N, T, symm2_core< N, M, true, T, core_ident<N, T, A> > >
symm(
    const letter_expr<M> sym1,
    const letter_expr<M> sym2,
    labeled_btensor<N, T, A> bt) {

    typedef expr< N, T, core_ident<N, T, A> > sub_expr_t;
    return symm(sym1, sym2, sub_expr_t(bt));
}


/** \brief Anti-symmetrization of a %tensor over two sets of indexes
    \tparam N Tensor order.
    \tparam M Number of indexes in each set.
    \tparam T Tensor element type.
    \tparam A Tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, bool A>
inline
expr< N, T, symm2_core< N, M, false, T, core_ident<N, T, A> > >
asymm(
    const letter_expr<M> sym1,
    const letter_expr<M> sym2,
    labeled_btensor<N, T, A> bt) {

    typedef expr< N, T, core_ident<N, T, A> > sub_expr_t;
    return asymm(sym1, sym2, sub_expr_t(bt));
}


/** \brief Symmetrization of a %tensor over two indexes
    \tparam N Tensor order.
    \tparam M Number of indexes in each set.
    \tparam T Tensor element type.
    \tparam A Tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A>
inline
expr< N, T, symm2_core< N, 1, true, T, core_ident<N, T, A> > >
symm(
    const letter &l1,
    const letter &l2,
    labeled_btensor<N, T, A> bt) {

    return symm(letter_expr<1>(l1), letter_expr<1>(l2), bt);
}


/** \brief Anti-symmetrization of a %tensor over two indexes
    \tparam N Tensor order.
    \tparam M Number of indexes in each set.
    \tparam T Tensor element type.
    \tparam A Tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A>
inline
expr< N, T, symm2_core< N, 1, false, T, core_ident<N, T, A> > >
asymm(
    const letter &l1,
    const letter &l2,
    labeled_btensor<N, T, A> bt) {

    return asymm(letter_expr<1>(l1), letter_expr<1>(l2), bt);
}


/** \brief Symmetrization of a %tensor over three indexes
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam A Tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A>
inline
expr< N, T, symm3_core< N, true, T, core_ident<N, T, A> > >
symm(
    const letter &l1,
    const letter &l2,
    const letter &l3,
    labeled_btensor<N, T, A> bt) {

    typedef expr< N, T, core_ident<N, T, A> > sub_expr_t;
    return symm(l1, l2, l3, sub_expr_t(bt));
}


/** \brief Anti-symmetrization of a %tensor over three indexes
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam A Tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A>
inline
expr< N, T, symm3_core< N, false, T, core_ident<N, T, A> > >
asymm(
    const letter &l1,
    const letter &l2,
    const letter &l3,
    labeled_btensor<N, T, A> bt) {

    typedef expr< N, T, core_ident<N, T, A> > sub_expr_t;
    return asymm(l1, l2, l3, sub_expr_t(bt));
}


} // namespace labeled_btensor_expr

using labeled_btensor_expr::symm;
using labeled_btensor_expr::asymm;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM_OPERATOR_H
