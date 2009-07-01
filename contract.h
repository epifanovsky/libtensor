#ifndef LIBTENSOR_CONTRACT_H
#define	LIBTENSOR_CONTRACT_H

#include "defs.h"
#include "exception.h"
#include "labeled_btensor.h"
#include "labeled_btensor_expr.h"
#include "labeled_btensor_expr_ident.h"
#include "letter.h"
#include "letter_expr.h"

namespace libtensor {

template<size_t N, size_t M, size_t K, typename T,
	typename Label, typename Expr1, typename Expr2>
class labeled_btensor_eval_contract;

/**	\brief Contraction operation for two arguments
	\tparam N Order of the first %tensor (a) less contraction degree.
	\tparam M Order of the second %tensor (b) less contraction degree.
	\tparam K Number of indexes contracted.
	\tparam Label Letter expression for contracted indexes.
	\tparam Expr1 First expression type (labeled_btensor_expr).
	\tparam Expr2 Second expression type (labeled_btensor_expr).

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K, typename T,
	typename Label, typename Expr1, typename Expr2>
class labeled_btensor_expr_contract {
public:
	typedef labeled_btensor_eval_contract<N, M, K, T, Label, Expr1, Expr2>
		eval_container_t;

private:
	Expr1 m_expr1; //!< First expression
	Expr2 m_expr2; //!< Second expression
	letter_expr<K, Label> m_contr; //!< Contracted indexes

public:
	labeled_btensor_expr_contract(const letter_expr<K, Label> &contr,
		const Expr1 &expr1, const Expr2 &expr2)
		: m_contr(contr), m_expr1(expr1), m_expr2(expr2) { }

};

/**	\brief Container for the evaluation of a contraction of two arguments
	\tparam N Order of the first %tensor (a) less contraction degree.
	\tparam M Order of the second %tensor (b) less contraction degree.
	\tparam K Number of indexes contracted.
	\tparam Label Letter expression for contracted indexes.
	\tparam Expr1 First expression type (labeled_btensor_expr).
	\tparam Expr2 Second expression type (labeled_btensor_expr).

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K, typename T,
	typename Label, typename Expr1, typename Expr2>
class labeled_btensor_eval_contract {
public:
	static const size_t k_ordera = N + K; //!< Order of the first %tensor
	static const size_t k_orderb = M + K; //!< Order of the second %tensor
	static const size_t k_orderc = N + M; //!< Order of the result

	//!	Contraction expression core type
	typedef labeled_btensor_expr_contract<N, M, K, T, Label, Expr1, Expr2>
		core_t;

	//!	Contraction expression type
	typedef labeled_btensor_expr<k_orderc, T, core_t> expression_t;

	//!	Number of %tensor arguments in the expression
	static const size_t k_narg_tensor = 0;

	//!	Number of %tensor operation arguments in the expression
	static const size_t k_narg_oper = 1;

private:
	expression_t &m_expr; //!< Contraction expression

public:
	/**	\brief Initializes the container with given expression and
			result recipient
	 **/
	template<typename LabelLhs>
	labeled_btensor_eval_contract(expression_t &expr,
		labeled_btensor<k_orderc, T, true, LabelLhs> &result)
		throw(exception);

	/**	\brief Returns a single %tensor argument
	 **/
	labeled_btensor_expr_arg_tensor<N + M, T> get_arg_tensor(size_t i) const
		throw(exception);

	/**	\brief Returns a single %tensor operation argument
	 **/
	labeled_btensor_expr_arg_oper<N + M, T> get_arg_oper(size_t i) const
		throw(exception);
};

template<size_t N, size_t M, size_t K, typename T,
	typename Label, typename Expr1, typename Expr2>
template<typename LabelLhs>
labeled_btensor_eval_contract<N, M, K, T, Label, Expr1, Expr2>::
labeled_btensor_eval_contract(expression_t &expr,
labeled_btensor<k_orderc, T, true, LabelLhs> &result) throw(exception)
: m_expr(expr) {

}

template<size_t N, size_t M, size_t K, typename T,
	typename Label, typename Expr1, typename Expr2>
labeled_btensor_expr_arg_tensor<N + M, T>
labeled_btensor_eval_contract<N, M, K, T, Label, Expr1, Expr2>::get_arg_tensor(
	size_t i) const	throw(exception) {

	throw_exc("labeled_btensor_expr_contract<N, M, K, T, Label, Expr1, Expr2>",
		"get_arg_tensor(size_t, letter_expr<N + M, Label>&)",
		"Invalid method to call");
}

template<size_t N, size_t M, size_t K, typename T,
	typename Label, typename Expr1, typename Expr2>
labeled_btensor_expr_arg_oper<N + M, T>
labeled_btensor_eval_contract<N, M, K, T, Label, Expr1, Expr2>::get_arg_oper(
	size_t i) const	throw(exception) {

	throw_exc("labeled_btensor_expr_contract<N, M, K, T, Label, Expr1, Expr2>",
		"get_arg_oper(size_t, letter_expr<N + M, Label>&)",
		"Not implemented");
}

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
labeled_btensor_expr< N + M - 2, T,
labeled_btensor_expr_contract< N - 1, M - 1, 1, T,
letter_expr< 1, letter_expr_ident >,
labeled_btensor_expr< N, T,
labeled_btensor_expr_ident< N, T, Assignable1, Label1 > >,
labeled_btensor_expr< M, T,
labeled_btensor_expr_ident< M, T, Assignable2, Label2 > >
> >
contract(const letter &let, labeled_btensor<N, T,Assignable1, Label1> bta,
	labeled_btensor<M, T, Assignable2, Label2> btb) {
	typedef letter_expr<1, letter_expr_ident> label_t;
	typedef labeled_btensor_expr_ident<N, T, Assignable1, Label1> id1_t;
	typedef labeled_btensor_expr<N, T, id1_t> expr1_t;
	typedef labeled_btensor_expr_ident<M, T, Assignable2, Label2> id2_t;
	typedef labeled_btensor_expr<M, T, id2_t> expr2_t;
	typedef labeled_btensor_expr_contract<N - 1, M - 1, 1, T,
		label_t, expr1_t, expr2_t> contract_t;
	typedef labeled_btensor_expr<N + M - 2, T, contract_t> expr_t;
	return expr_t(contract_t(
		label_t(let), expr1_t(id1_t(bta)), expr2_t(id2_t(btb))));
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
labeled_btensor_expr< N + M - 2 * K, T,
labeled_btensor_expr_contract< N - K, M - K, K, T,
letter_expr< K, Contr >,
labeled_btensor_expr< N, T,
	labeled_btensor_expr_ident< N, T, Assignable1, Label1 > >,
labeled_btensor_expr< M, T,
	labeled_btensor_expr_ident< M, T, Assignable2, Label2 > >
> >
contract(const letter_expr<K, Contr> &contr,
	labeled_btensor<N, T, Assignable1, Label1> bta,
	labeled_btensor<M, T, Assignable2, Label2> btb) {

	typedef letter_expr<K, Contr> label_t;
	typedef labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, Assignable1, Label1> > expr1_t;
	typedef labeled_btensor_expr<M, T,
		labeled_btensor_expr_ident<M, T, Assignable2, Label2> > expr2_t;
	typedef labeled_btensor_expr_contract<N - K, M - K, K, T,
		label_t, expr1_t, expr2_t> contract_t;
	typedef labeled_btensor_expr<N + M - 2 * K, T, contract_t> expr_t;
	return expr_t(contract_t(contr, expr1_t(bta), expr2_t(btb)));
}

template<size_t K, size_t N, size_t M, typename T, typename Contr,
	typename Expr1, bool Assignable2, typename Label2>
labeled_btensor_expr<N + M - 2 * K, T,
labeled_btensor_expr_contract<N - K, M - K, K, T,
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
	typedef labeled_btensor_expr_contract<N - K, M - K, K, T,
		label_t, expr1_t, expr2_t> contract_t;
	typedef labeled_btensor_expr<N + M - 2 * K, T, contract_t> expr_t;
	return expr_t(contract_t(contr, bta, expr2_t(btb)));
}

template<size_t K, size_t N, size_t M, typename T, typename Contr,
	bool Assignable1, typename Label1, typename Expr2>
labeled_btensor_expr<N + M - 2 * K, T,
labeled_btensor_expr_contract<N - K, M - K, K, T,
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
	typedef labeled_btensor_expr_contract<N - K, M - K, K, T,
		label_t, expr1_t, expr2_t> contract_t;
	typedef labeled_btensor_expr<N + M - 2 * K, T, contract_t> expr_t;
	return expr_t(contract_t(contr, expr1_t(bta), btb));
}

} // namespace libtensor

#endif // LIBTENSOR_CONTRACT_H

