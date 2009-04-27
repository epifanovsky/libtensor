#ifndef LIBTENSOR_CONTRACT_H
#define	LIBTENSOR_CONTRACT_H

#include "defs.h"
#include "exception.h"
#include "labeled_btensor.h"
#include "labeled_btensor_expr.h"
#include "letter.h"
#include "letter_expr.h"

namespace libtensor {

/**	\brief Contraction operation for two arguments
	\tparam Expr1 First expression
	\tparam Expr2 Second expression

	\ingroup libtensor_btensor_expr
 **/
template<typename Expr1, typename Expr2>
class labeled_btensor_expr_op_contract {
};

template<size_t N, size_t M, typename T, typename Traits1, typename Label1,
	typename Traits2, typename Label2>
void
contract(letter &, labeled_btensor<N, T, Traits1, Label1>,
	labeled_btensor<M, T, Traits2, Label2>) {
}

/**	\brief Contraction of two tensors
	\tparam K Number of contracted indexes.
	\tparam N Order of the first tensor.
	\tparam M Order of the second tensor.
	\tparam T Tensor element type.
	\tparam Contr Contraction letter expression.
	\tparam Traits1 Traits of the first tensor.
	\tparam Label1 Label of the first tensor.
	\tparam Traits2 Traits of the second tensor.
	\tparam Label2 Label of the second tensor.

	\ingroup libtensor_btensor_expr
 **/
template<size_t K, size_t N, size_t M, typename T, typename Contr,
	typename Traits1, typename Label1, typename Traits2, typename Label2>
//labeled_btensor_expr<N + M - 2 * K, T,
//labeled_btensor_expr_op<N, T, 2, labeled_btensor_expr_op_contract<
//labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsL, LabelL> >,
//labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsL, LabelL> >
//>, void, void
//> >
void
contract(letter_expr<K, Contr>, labeled_btensor<N, T, Traits1, Label1>,
	labeled_btensor<M, T, Traits2, Label2>) {
}

} // namespace libtensor

#endif // LIBTENSOR_CONTRACT_H

