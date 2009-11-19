#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM_OPERATOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM_OPERATOR_H

#include "symm2_core.h"
#include "symm2_eval.h"

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Symmetrization of an expression over two indexes
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam SubCore Sub-expression core.

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename SubCore>
inline
expr< N, T, symm2_core<N, true, T, SubCore> >
symm(
	const letter_expr<2> sym,
	expr<N, T, SubCore> subexpr) {

	typedef expr<N, T, SubCore> sub_expr_t;
	typedef symm2_core<N, true, T, SubCore> core_t;
	typedef expr<N, T, core_t> expr_t;
	return expr_t(core_t(sym, subexpr));
}


/**	\brief Symmetrization of a %tensor over two indexes
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam A Tensor assignable.

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A>
inline
expr< N, T, symm2_core< N, true, T, core_ident<N, T, A> > >
symm(
	const letter_expr<2> sym,
	labeled_btensor<N, T, A> bt) {

	typedef expr< N, T, core_ident<N, T, A> > sub_expr_t;
	return symm(sym, sub_expr_t(bt));
}


/**	\brief Anti-symmetrization of an expression over two indexes
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam SubCore Sub-expression core.

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename SubCore>
inline
expr< N, T, symm2_core<N, false, T, SubCore> >
asymm(
	const letter_expr<2> sym,
	expr<N, T, SubCore> subexpr) {

	typedef expr<N, T, SubCore> sub_expr_t;
	typedef symm2_core<N, false, T, SubCore> core_t;
	typedef expr<N, T, core_t> expr_t;
	return expr_t(core_t(sym, subexpr));
}


/**	\brief Anti-symmetrization of a %tensor over two indexes
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam A Tensor assignable.

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A>
inline
expr< N, T, symm2_core< N, false, T, core_ident<N, T, A> > >
asymm(
	const letter_expr<2> sym,
	labeled_btensor<N, T, A> bt) {

	typedef expr< N, T, core_ident<N, T, A> > sub_expr_t;
	return symm(sym, sub_expr_t(bt));
}


} // namespace labeled_btensor_expr

using labeled_btensor_expr::symm;
using labeled_btensor_expr::asymm;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM_OPERATOR_H
