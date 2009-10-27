#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_SYM_CONTRACT_OPERATOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_SYM_CONTRACT_OPERATOR_H

#include "sym_contract_core.h"
#include "sym_contract_eval.h"

namespace libtensor {
namespace labeled_btensor_expr {

/**	\brief Symmetrized contraction of two expressions over multiple indexes
		(symmetrization of two indexes)
	\tparam K Number of contracted indexes.
	\tparam N Order of the first %tensor.
	\tparam M Order of the second %tensor.
	\tparam T Tensor element type.
	\tparam E1 First expression.
	\tparam E2 Second expression.

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t K, size_t N, size_t M, typename T, typename E1, typename E2>
inline
expr<N + M - 2 * K, T, sym_contract_core<N - K, M - K, K, T,
	expr<N, T, E1>,
	expr<M, T, E2>
> >
sym_contract(
	const letter_expr<2> sym,
	const letter_expr<K> contr,
	expr<N, T, E1> bta,
	expr<M, T, E2> btb) {

	typedef expr<N, T, E1> expr1_t;
	typedef expr<M, T, E2> expr2_t;
	typedef sym_contract_core<N - K, M - K, K, T, expr1_t, expr2_t> core_t;
	typedef expr<N + M - 2 * K, T, core_t> expr_t;
	return expr_t(core_t(sym, contr, bta, btb));
}


/**	\brief Symmetrized contraction of two expressions over one %index
		(symmetrization of two indexes)
	\tparam N Order of the first %tensor.
	\tparam M Order of the second %tensor.
	\tparam T Tensor element type.
	\tparam E1 First expression.
	\tparam E2 Second expression.

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, typename E1, typename E2>
inline
expr<N + M - 2, T, sym_contract_core<N - 1, M - 1, 1, T,
	expr<N, T, E1>,
	expr<M, T, E2>
> >
sym_contract(
	const letter_expr<2> sym,
	const letter &let,
	expr<N, T, E1> bta,
	expr<M, T, E2> btb) {

	return sym_contract(sym, letter_expr<1>(let), bta, btb);
}


/**	\brief Symmetrized contraction of two tensors over multiple indexes
		(symmetrization of two indexes)
	\tparam K Number of contracted indexes.
	\tparam N Order of the first tensor.
	\tparam M Order of the second tensor.
	\tparam T Tensor element type.
	\tparam A1 First %tensor assignable.
	\tparam A2 Second %tensor assignable.

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t K, size_t N, size_t M, typename T, bool A1, bool A2>
inline
expr<N + M - 2 * K, T, sym_contract_core<N - K, M - K, K, T,
	expr< N, T, core_ident<N, T, A1> >,
	expr< M, T, core_ident<M, T, A2> >
> >
sym_contract(
	const letter_expr<2> sym,
	const letter_expr<K> contr,
	labeled_btensor<N, T, A1> bta,
	labeled_btensor<M, T, A2> btb) {

	typedef expr< N, T, core_ident<N, T, A1> > expr1_t;
	typedef expr< M, T, core_ident<M, T, A2> > expr2_t;
	return sym_contract(sym, contr, expr1_t(bta), expr2_t(btb));
}


/**	\brief Symmetrized contraction of two tensors over one index
		(symmetrization of two indexes)
	\tparam N Order of the first %tensor.
	\tparam M Order of the second %tensor.
	\tparam T Tensor element type.
	\tparam A1 First %tensor assignable.
	\tparam A2 Second %tensor assignable.

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, bool A1, bool A2>
inline
expr<N + M - 2, T, sym_contract_core<N - 1, M - 1, 1, T,
	expr< N, T, core_ident<N, T, A1> >,
	expr< M, T, core_ident<M, T, A2> >
> >
sym_contract(
	const letter_expr<2> sym,
	const letter &let,
	labeled_btensor<N, T, A1> bta,
	labeled_btensor<M, T, A2> btb) {

	return sym_contract(sym, letter_expr<1>(let), bta, btb);
}


} // namespace labeled_btensor_expr

using labeled_btensor_expr::sym_contract;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_SYM_CONTRACT_OPERATOR_H
