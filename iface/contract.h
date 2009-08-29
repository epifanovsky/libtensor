#ifndef LIBTENSOR_CONTRACT_H
#define	LIBTENSOR_CONTRACT_H

#include "defs.h"
#include "exception.h"
#include "core/permutation_builder.h"
#include "btod/btod_contract2.h"
#include "labeled_btensor.h"
#include "labeled_btensor_expr.h"
#include "labeled_btensor_expr_ident.h"
#include "letter.h"
#include "letter_expr.h"

#include "contract/core_contract.h"
#include "contract/eval_contract.h"

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Contraction of two expressions over multiple indexes
	\tparam K Number of contracted indexes.
	\tparam N Order of the first %tensor.
	\tparam M Order of the second %tensor.
	\tparam T Tensor element type.
	\tparam E1 First expression.
	\tparam E2 Second expression.

	\ingroup libtensor_btensor_expr
 **/
template<size_t K, size_t N, size_t M, typename T, typename E1, typename E2>
expr<N + M - 2 * K, T, core_contract<N - K, M - K, K, T,
	expr<N, T, E1>,
	expr<M, T, E2>
> >
inline contract(
	const letter_expr<K> contr,
	expr<N, T, E1> bta,
	expr<M, T, E2> btb) {

	typedef expr<N, T, E1> expr1_t;
	typedef expr<M, T, E2> expr2_t;
	typedef core_contract<N - K, M - K, K, T, expr1_t, expr2_t> core_t;
	typedef expr<N + M - 2 * K, T, core_t> expr_t;
	return expr_t(core_t(contr, bta, btb));
}


/**	\brief Contraction of two expressions over one %index
	\tparam N Order of the first %tensor.
	\tparam M Order of the second %tensor.
	\tparam T Tensor element type.
	\tparam E1 First expression.
	\tparam E2 Second expression.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T, typename E1, typename E2>
expr<N + M - 2, T, core_contract<N - 1, M - 1, 1, T,
	expr<N, T, E1>,
	expr<M, T, E2>
> >
inline contract(
	const letter &let,
	expr<N, T, E1> bta,
	expr<M, T, E2> btb) {

	return contract(letter_expr<1>(let), bta, btb);
}


/**	\brief Contraction of a %tensor and an expression over multiple indexes
	\tparam K Number of contracted indexes.
	\tparam N Order of the first %tensor.
	\tparam M Order of the second %tensor.
	\tparam T Tensor element type.
	\tparam A1 First %tensor assignable.
	\tparam E2 Second expression.

	\ingroup libtensor_btensor_expr
 **/
template<size_t K, size_t N, size_t M, typename T, bool A1, typename E2>
expr<N + M - 2 * K, T, core_contract<N - K, M - K, K, T,
	expr< N, T, core_ident<N, T, A1> >,
	expr< M, T, E2 >
> >
inline contract(
	const letter_expr<K> contr,
	labeled_btensor<N, T, A1> bta,
	expr<M, T, E2> btb) {

	typedef expr< N, T, core_ident<N, T, A1> > expr1_t;
	return contract(contr, expr1_t(bta), btb);
}


/**	\brief Contraction of a %tensor and an expression over one %index
	\tparam N Order of the first %tensor.
	\tparam M Order of the second %tensor.
	\tparam T Tensor element type.
	\tparam A1 First %tensor assignable.
	\tparam E2 Second expression.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T, bool A1, typename E2>
expr<N + M - 2, T, core_contract<N - 1, M - 1, 1, T,
	expr< N, T, core_ident<N, T, A1> >,
	expr< M, T, E2 >
> >
inline contract(
	const letter &let,
	labeled_btensor<N, T, A1> bta,
	expr<M, T, E2> btb) {

	return contract(letter_expr<1>(let), bta, btb);
}


/**	\brief Contraction of an expression and a %tensor over multiple indexes
	\tparam K Number of contracted indexes.
	\tparam N Order of the first %tensor.
	\tparam M Order of the second %tensor.
	\tparam T Tensor element type.
	\tparam E1 First expression.
	\tparam A2 Second %tensor assignable.

	\ingroup libtensor_btensor_expr
 **/
template<size_t K, size_t N, size_t M, typename T, typename E1, bool A2>
expr<N + M - 2 * K, T, core_contract<N - K, M - K, K, T,
	expr< N, T, E1 >,
	expr< M, T, core_ident<M, T, A2> >
> >
inline contract(
	const letter_expr<K> contr,
	expr<N, T, E1> bta,
	labeled_btensor<M, T, A2> btb) {

	typedef expr< M, T, core_ident<M, T, A2> > expr2_t;
	return contract(contr, bta, expr2_t(btb));
}


/**	\brief Contraction of an expression and a %tensor over one %index
	\tparam N Order of the first %tensor.
	\tparam M Order of the second %tensor.
	\tparam T Tensor element type.
	\tparam E1 First expression.
	\tparam A2 Second %tensor assignable.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T, typename E1, bool A2>
expr<N + M - 2, T, core_contract<N - 1, M - 1, 1, T,
	expr< N, T, E1 >,
	expr< M, T, core_ident<M, T, A2> >
> >
inline contract(
	const letter &let,
	expr<N, T, E1> bta,
	labeled_btensor<M, T, A2> btb) {

	return contract(letter_expr<1>(let), bta, btb);
}


/**	\brief Contraction of two tensors over multiple indexes
	\tparam K Number of contracted indexes.
	\tparam N Order of the first tensor.
	\tparam M Order of the second tensor.
	\tparam T Tensor element type.
	\tparam A1 First %tensor assignable.
	\tparam A2 Second %tensor assignable.

	\ingroup libtensor_btensor_expr
 **/
template<size_t K, size_t N, size_t M, typename T, bool A1, bool A2>
expr<N + M - 2 * K, T, core_contract<N - K, M - K, K, T,
	expr< N, T, core_ident<N, T, A1> >,
	expr< M, T, core_ident<M, T, A2> >
> >
inline contract(
	const letter_expr<K> contr,
	labeled_btensor<N, T, A1> bta,
	labeled_btensor<M, T, A2> btb) {

	typedef expr< N, T, core_ident<N, T, A1> > expr1_t;
	typedef expr< M, T, core_ident<M, T, A2> > expr2_t;
	return contract(contr, expr1_t(bta), expr2_t(btb));
}


/**	\brief Contraction of two tensors over one index
	\tparam N Order of the first %tensor.
	\tparam M Order of the second %tensor.
	\tparam T Tensor element type.
	\tparam A1 First %tensor assignable.
	\tparam A2 Second %tensor assignable.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T, bool A1, bool A2>
expr<N + M - 2, T, core_contract<N - 1, M - 1, 1, T,
	expr< N, T, core_ident<N, T, A1> >,
	expr< M, T, core_ident<M, T, A2> >
> >
inline contract(
	const letter &let,
	labeled_btensor<N, T, A1> bta,
	labeled_btensor<M, T, A2> btb) {

	return contract(letter_expr<1>(let), bta, btb);
}


} // namespace labeled_btensor_expr

using labeled_btensor_expr::contract;

} // namespace libtensor

#endif // LIBTENSOR_CONTRACT_H

