#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_OPERATOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_OPERATOR_H

#include "diag_core.h"
#include "diag_eval.h"

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Extraction of a general %tensor diagonal (expression)
	\tparam N Tensor order.
	\tparam M Diagonal order.
	\tparam T Tensor element type.
	\tparam E1 Sub-expression core.

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, typename E1>
inline
expr< N - M + 1, T, diag_core<N, M, T, E1> >
diag(
	const letter &let_diag,
	const letter_expr<M> lab_diag,
	expr<N, T, E1> subexpr) {

	typedef expr<N, T, E1> sub_expr_t;
	typedef diag_core<N, M, T, E1> core_t;
	typedef expr<N - M + 1, T, core_t> expr_t;
	return expr_t(core_t(sym, subexpr));
}


/**	\brief Extraction of a general %tensor diagonal (%tensor)
	\tparam N Tensor order.
	\tparam M Diagonal order.
	\tparam T Tensor element type.
	\tparam A Tensor assignable.

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, bool A>
inline
expr< N - M + 1, T, diag_core< N, M, T, core_ident<N, T, A> > >
diag(
	const letter &let_diag,
	const letter_expr<M> lab_diag,
	labeled_btensor<N, T, A> bt) {

	typedef expr< N, T, core_ident<N, T, A> > sub_expr_t;
	return diag(let_diag, lab_diag, sub_expr_t(bt));
}


} // namespace labeled_btensor_expr

using labeled_btensor_expr::diag;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_OPERATOR_H
