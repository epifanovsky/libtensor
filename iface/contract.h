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

namespace libtensor {

namespace labeled_btensor_expr {

template<size_t N, size_t M, size_t K, typename T,
	typename Label, typename Expr1, typename Expr2>
class eval_contract;

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
class core_contract {
private:
	static const char *k_clazz; //!< Class name

public:
	typedef eval_contract<N, M, K, T, Label, Expr1, Expr2>
		eval_container_t; //!< Evaluating container type

private:
	Expr1 m_expr1; //!< First expression
	Expr2 m_expr2; //!< Second expression
	letter_expr<K, Label> m_contr; //!< Contracted indexes
	const letter *m_defout[N + M]; //!< Default output label

public:
	core_contract(const letter_expr<K, Label> &contr,
		const Expr1 &expr1, const Expr2 &expr2);

	Expr1 &get_expr_1() { return m_expr1; }
	Expr2 &get_expr_2() { return m_expr2; }

	/**	\brief Returns whether the %tensor's label contains a %letter
	 **/
	bool contains(const letter &let) const;

	/**	\brief Returns the %index of a %letter in the %tensor's label
	 **/
	size_t index_of(const letter &let) const throw(exception);

	/**	\brief Returns the %letter at a given position in
			the %tensor's label
	 **/
	const letter &letter_at(size_t i) const throw(exception);

};

template<size_t N, size_t M, size_t K, typename T, typename Label,
	typename Expr1, typename Expr2>
const char *core_contract<N, M, K, T, Label, Expr1, Expr2>::
	k_clazz = "core_contract<N, M, K, T, Label, Expr1, Expr2>";


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
class eval_contract {
public:
	static const size_t k_ordera = N + K; //!< Order of the first %tensor
	static const size_t k_orderb = M + K; //!< Order of the second %tensor
	static const size_t k_orderc = N + M; //!< Order of the result

	//!	Contraction expression core type
	typedef core_contract<N, M, K, T, Label, Expr1, Expr2> core_t;

	//!	Contraction expression type
	typedef expr<k_orderc, T, core_t> expression_t;

	//!	Number of arguments in the expression
	template<typename Tag, int Dummy = 0>
	struct narg {
		static const size_t k_narg = 0;
	};

private:
	expression_t &m_expr; //!< Contraction expression
	contraction2<N, M, K> m_contr; //!< Contraction
	//btod_contract2<N, M, K> m_op; //!< Contraction operation

public:
	/**	\brief Initializes the container with given expression and
			result recipient
	 **/
	template<typename LabelLhs>
	eval_contract(expression_t &expr,
		labeled_btensor<k_orderc, T, true, LabelLhs> &result)
		throw(exception);

	template<typename Tag>
	arg<N + M, T, Tag> get_arg(const Tag &tag, size_t i) const
		throw(exception);

	/**	\brief Returns a single %tensor operation argument
	 **/
	arg<N + M, T, oper_tag> get_arg(const oper_tag &tag, size_t i) const
		throw(exception);

private:
	template<typename LabelLhs>
	static contraction2<N, M, K> mk_contr(expression_t &expr,
		labeled_btensor<k_orderc, T, true, LabelLhs> &result)
		throw(exception);
};

template<size_t N, size_t M, size_t K, typename T, typename Label,
	typename Expr1, typename Expr2>
template<int Dummy>
struct eval_contract<N, M, K, T, Label, Expr1, Expr2>::narg<oper_tag, Dummy> {
	static const size_t k_narg = 1;
};

template<size_t N, size_t M, size_t K, typename T,
	typename Label, typename Expr1, typename Expr2>
core_contract<N, M, K, T, Label, Expr1, Expr2>::
core_contract(const letter_expr<K, Label> &contr,
	const Expr1 &expr1, const Expr2 &expr2)
	: m_contr(contr), m_expr1(expr1), m_expr2(expr2) {

	static const char *method = "core_contract("
		"const letter_expr<K, Label>&, const Expr1&, const Expr2&)";

	for(size_t i = 0; i < K; i++) {
		const letter &l = contr.letter_at(i);
		if(!expr1.contains(l) || !expr2.contains(l)) {
			throw_exc(k_clazz, method,
				"Contracted index is absent from an argument");
		}
	}

	size_t j = 0;
	for(size_t i = 0; i < N + K; i++) {
		const letter &l = expr1.letter_at(i);
		if(!contr.contains(l)) {
			if(expr2.contains(l)) {
				throw_exc(k_clazz, method,
					"Duplicate uncontracted index");
			} else {
				m_defout[j++] = &l;
			}
		}
	}
	for(size_t i = 0; i < M + K; i++) {
		const letter &l = expr2.letter_at(i);
		if(!contr.contains(l)) {
			if(expr1.contains(l)) {
				throw_exc(k_clazz, method,
					"Duplicate uncontracted index");
			} else {
				m_defout[j++] = &l;
			}
		}
	}
}

template<size_t N, size_t M, size_t K, typename T,
	typename Label, typename Expr1, typename Expr2>
inline bool core_contract<N, M, K, T, Label, Expr1, Expr2>::
	contains(const letter &let) const {

	for(register size_t i = 0; i < N + M; i++) {
		if(m_defout[i] == &let) return true;
	}
	return false;
}

template<size_t N, size_t M, size_t K, typename T,
	typename Label, typename Expr1, typename Expr2>
inline size_t core_contract<N, M, K, T, Label, Expr1, Expr2>::
	index_of(const letter &let) const throw(exception) {

	static const char *method = "index_of(const letter&)";

	for(register size_t i = 0; i < N + M; i++) {
		if(m_defout[i] == &let) return i;
	}

	throw_exc(k_clazz, method, "Expression doesn't contain the letter");
}

template<size_t N, size_t M, size_t K, typename T,
	typename Label, typename Expr1, typename Expr2>
inline const letter&core_contract<N, M, K, T, Label, Expr1, Expr2>::letter_at(
	size_t i) const throw(exception) {

	static const char *method = "letter_at(size_t)";

	if(i >= N + M) throw_exc(k_clazz, method, "Index out of bounds");
	return *(m_defout[i]);
}


template<size_t N, size_t M, size_t K, typename T,
	typename Label, typename Expr1, typename Expr2>
template<typename LabelLhs>
eval_contract<N, M, K, T, Label, Expr1, Expr2>::eval_contract(
	expression_t &expr,
	labeled_btensor<k_orderc, T, true, LabelLhs> &result) throw(exception)
: m_expr(expr), m_contr(mk_contr(expr, result))/*, m_op(m_contr)*/ {

}

template<size_t N, size_t M, size_t K, typename T,
	typename Label, typename Expr1, typename Expr2>
template<typename Tag>
arg<N + M, T, Tag> eval_contract<N, M, K, T, Label, Expr1, Expr2>::get_arg(
	const Tag &tag, size_t i) const throw(exception) {

	throw_exc("eval_contract<N, M, K, T, Label, Expr1, Expr2>",
		"get_arg(const Tag&, letter_expr<N + M, Label>&)",
		"Invalid method to call");
}

template<size_t N, size_t M, size_t K, typename T,
	typename Label, typename Expr1, typename Expr2>
arg<N + M, T, oper_tag> eval_contract<N, M, K, T, Label, Expr1, Expr2>::get_arg(
	const oper_tag &tag, size_t i) const throw(exception) {

	throw_exc("eval_contract<N, M, K, T, Label, Expr1, Expr2>",
		"get_arg(const oper_tag&, letter_expr<N + M, Label>&)",
		"Not implemented");
}

template<size_t N, size_t M, size_t K, typename T,
	typename Label, typename Expr1, typename Expr2>
template<typename LabelLhs>
contraction2<N, M, K>
eval_contract<N, M, K, T, Label, Expr1, Expr2>::mk_contr(
	expression_t &expr,
	labeled_btensor<k_orderc, T, true, LabelLhs> &result) throw(exception) {

	size_t seq1[N + M], seq2[N + M];
	for(size_t i = 0; i < N + M; i++) {
		seq1[i] = i;
		seq2[i] = expr.index_of(result.letter_at(i));
	}
	permutation_builder<N + M> permc(seq1, seq2);
	contraction2<N, M, K> contr(permc.get_perm());

	for(size_t i = 0; i < N + K; i++) {
		const letter &l = expr.get_core().get_expr_1().letter_at(i);
		if(expr.get_core().get_expr_2().contains(l)) {
			contr.contract(i, expr.get_core().get_expr_2().index_of(l));
		}
	}

	return contr;
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
expr<N + M - 2, T, core_contract<N - 1, M - 1, 1, T,
	letter_expr<1, letter_expr_ident>,
	expr<N, T, core_ident<N, T, Assignable1, Label1> >,
	expr<M, T, core_ident<M, T, Assignable2, Label2> >
> >
inline contract(
	const letter &let,
	labeled_btensor<N, T,Assignable1, Label1> bta,
	labeled_btensor<M, T, Assignable2, Label2> btb) {

	typedef letter_expr<1, letter_expr_ident> label_t;
	typedef core_ident<N, T, Assignable1, Label1> id1_t;
	typedef expr<N, T, id1_t> expr1_t;
	typedef core_ident<M, T, Assignable2, Label2> id2_t;
	typedef expr<M, T, id2_t> expr2_t;
	typedef core_contract<N - 1, M - 1, 1, T, label_t, expr1_t, expr2_t>
		contract_t;
	typedef expr<N + M - 2, T, contract_t> expr_t;
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
expr<N + M - 2 * K, T, core_contract<N - K, M - K, K, T,
	letter_expr<K, Contr>,
	expr<N, T, core_ident< N, T, Assignable1, Label1> >,
	expr<M, T, core_ident< M, T, Assignable2, Label2> >
> >
inline contract(
	const letter_expr<K, Contr> &contr,
	labeled_btensor<N, T, Assignable1, Label1> bta,
	labeled_btensor<M, T, Assignable2, Label2> btb) {

	typedef letter_expr<K, Contr> label_t;
	typedef expr<N, T, core_ident<N, T, Assignable1, Label1> > expr1_t;
	typedef expr<M, T, core_ident<M, T, Assignable2, Label2> > expr2_t;
	typedef core_contract<N - K, M - K, K, T, label_t, expr1_t, expr2_t>
		contract_t;
	typedef expr<N + M - 2 * K, T, contract_t> expr_t;
	return expr_t(contract_t(contr, expr1_t(bta), expr2_t(btb)));
}

template<size_t K, size_t N, size_t M, typename T, typename Contr,
	typename Expr1, bool Assignable2, typename Label2>
expr<N + M - 2 * K, T, core_contract<N - K, M - K, K, T,
	letter_expr<K, Contr>,
	expr<N, T, Expr1>,
	expr<M, T, core_ident<M, T, Assignable2, Label2> >
> >
inline contract(
	const letter_expr<K, Contr> &contr,
	expr<N, T, Expr1> bta,
	labeled_btensor<M, T, Assignable2, Label2> btb) {

	typedef letter_expr<K, Contr> label_t;
	typedef expr<N, T, Expr1> expr1_t;
	typedef expr<M, T, core_ident<M, T, Assignable2, Label2> > expr2_t;
	typedef core_contract<N - K, M - K, K, T, label_t, expr1_t, expr2_t>
		contract_t;
	typedef expr<N + M - 2 * K, T, contract_t> expr_t;
	return expr_t(contract_t(contr, bta, expr2_t(btb)));
}

template<size_t K, size_t N, size_t M, typename T, typename Contr,
	bool Assignable1, typename Label1, typename Expr2>
expr<N + M - 2 * K, T, core_contract<N - K, M - K, K, T,
	letter_expr<K, Contr>,
	expr<N, T, core_ident<N, T, Assignable1, Label1> >,
	expr<M, T, Expr2>
> >
inline contract(
	const letter_expr<K, Contr> &contr,
	labeled_btensor<N, T,Assignable1, Label1> bta,
	expr<M, T, Expr2> btb) {

	typedef letter_expr<K, Contr> label_t;
	typedef expr<N, T, core_ident<N, T, Assignable1, Label1> > expr1_t;
	typedef expr<M, T, Expr2> expr2_t;
	typedef core_contract<N - K, M - K, K, T, label_t, expr1_t, expr2_t>
		contract_t;
	typedef expr<N + M - 2 * K, T, contract_t> expr_t;
	return expr_t(contract_t(contr, expr1_t(bta), btb));
}

} // namespace labeled_btensor_expr

using labeled_btensor_expr::contract;

} // namespace libtensor

#endif // LIBTENSOR_CONTRACT_H

