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
	\tparam Label Letter expression for contracted indexes
	\tparam Expr1 First expression
	\tparam Expr2 Second expression

	\ingroup libtensor_btensor_expr
 **/
template<typename Label, typename Expr1, typename Expr2>
class labeled_btensor_expr_op_contract {
private:
	Label m_contr; //!< Contracted indexes
	Expr1 m_expr1; //!< First expression
	Expr2 m_expr2; //!< Second expression

public:
	labeled_btensor_expr_op_contract(const Label &contr,
		const Expr1 &expr1, const Expr2 &expr2) :
		m_contr(contr), m_expr1(expr1), m_expr2(expr2) { }
};

/**	\brief Contraction of two tensors over one index
	\tparam N Order of the first %tensor.
	\tparam M Order of the second %tensor.
	\tparam Assignable1 Whether the first %tensor is assignable.
	\tparam Label1 Label of the first %tensor.
	\tparam Assignable2 Whether the second %tensor is assignable.
	\tparam Label2 Label of the second %tensor.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T, bool Assignable1, typename Label1,
	bool Assignable2, typename Label2>
labeled_btensor_expr<N + M - 2, T,
labeled_btensor_expr_op_contract<
	letter_expr<1, letter_expr_ident>,
	labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, Assignable1, Label1> >,
	labeled_btensor_expr<M, T,
		labeled_btensor_expr_ident<M, T, Assignable2, Label2> >
	>
>
contract(const letter &let, labeled_btensor<N, T,Assignable1, Label1> bta,
	labeled_btensor<M, T, Assignable2, Label2> btb) {
	typedef letter_expr<1, letter_expr_ident> label_t;
	typedef labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, Assignable1, Label1> > expr1_t;
	typedef labeled_btensor_expr<M, T,
		labeled_btensor_expr_ident<M, T, Assignable2, Label2> > expr2_t;
	typedef labeled_btensor_expr_op_contract<label_t, expr1_t, expr2_t>
		contract_t;
	typedef labeled_btensor_expr<N + M - 2, T, contract_t> expr_t;
	return expr_t(contract_t(label_t(let), expr1_t(bta), expr2_t(btb)));
}

/**	\brief Contraction of two tensors over multiple indexes
	\tparam K Number of contracted indexes.
	\tparam N Order of the first tensor.
	\tparam M Order of the second tensor.
	\tparam T Tensor element type.
	\tparam Contr Contraction letter expression.
	\tparam Assignable1 Whether the first %tensor is assignable.
	\tparam Label1 Label of the first %tensor.
	\tparam Assignable2 Whether the second %tensor is assignable.
	\tparam Label2 Label of the second tensor.

	\ingroup libtensor_btensor_expr
 **/
template<size_t K, size_t N, size_t M, typename T, typename Contr,
	bool Assignable1, typename Label1, bool Assignable2, typename Label2>
labeled_btensor_expr<N + M - 2 * K, T,
labeled_btensor_expr_op_contract<
	letter_expr<K, Contr>,
	labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, Assignable1, Label1> >,
	labeled_btensor_expr<M, T,
		labeled_btensor_expr_ident<M, T, Assignable2, Label2> >
	>
>
contract(const letter_expr<K, Contr> &contr,
	labeled_btensor<N, T, Assignable1, Label1> bta,
	labeled_btensor<M, T, Assignable2, Label2> btb) {

	typedef letter_expr<K, Contr> label_t;
	typedef labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, Assignable1, Label1> > expr1_t;
	typedef labeled_btensor_expr<M, T,
		labeled_btensor_expr_ident<M, T, Assignable2, Label2> > expr2_t;
	typedef labeled_btensor_expr_op_contract<label_t, expr1_t, expr2_t>
		contract_t;
	typedef labeled_btensor_expr<N + M - 2 * K, T, contract_t> expr_t;
	return expr_t(contract_t(contr, expr1_t(bta), expr2_t(btb)));
}

template<size_t K, size_t N, size_t M, typename T, typename Contr,
	typename Expr1, bool Assignable2, typename Label2>
labeled_btensor_expr<N + M - 2 * K, T,
labeled_btensor_expr_op_contract<
	letter_expr<K, Contr>,
	labeled_btensor_expr<N, T, Expr1>,
	labeled_btensor_expr<M, T,
		labeled_btensor_expr_ident<M, T, Assignable2, Label2> >
	>
>
contract(const letter_expr<K, Contr> &contr,
	labeled_btensor_expr<N, T, Expr1> bta,
	labeled_btensor<M, T, Assignable2, Label2> btb) {
	typedef letter_expr<K, Contr> label_t;
	typedef labeled_btensor_expr<N, T, Expr1> expr1_t;
	typedef labeled_btensor_expr<M, T,
		labeled_btensor_expr_ident<M, T, Assignable2, Label2> > expr2_t;
	typedef labeled_btensor_expr_op_contract<label_t, expr1_t, expr2_t>
		contract_t;
	typedef labeled_btensor_expr<N + M - 2 * K, T, contract_t> expr_t;
	return expr_t(contract_t(contr, bta, expr2_t(btb)));
}

template<size_t K, size_t N, size_t M, typename T, typename Contr,
	bool Assignable1, typename Label1, typename Expr2>
labeled_btensor_expr<N + M - 2 * K, T,
labeled_btensor_expr_op_contract<
	letter_expr<K, Contr>,
	labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, Assignable1, Label1> >,
	labeled_btensor_expr<M, T, Expr2>
	>
>
contract(const letter_expr<K, Contr> &contr,
	labeled_btensor<N, T,Assignable1, Label1> bta,
	labeled_btensor_expr<M, T, Expr2> btb) {
	typedef letter_expr<K, Contr> label_t;
	typedef labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, Assignable1, Label1> > expr1_t;
	typedef labeled_btensor_expr<M, T, Expr2> expr2_t;
	typedef labeled_btensor_expr_op_contract<label_t, expr1_t, expr2_t>
		contract_t;
	typedef labeled_btensor_expr<N + M - 2 * K, T, contract_t> expr_t;
	return expr_t(contract_t(contr, expr1_t(bta), btb));
}

} // namespace libtensor

#endif // LIBTENSOR_CONTRACT_H

