#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_EVALFUNCTOR_CONTRACT_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_EVALFUNCTOR_CONTRACT_H

#include "iface/expr/anon_eval.h"
#include "core_contract.h"

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, size_t M, size_t K, size_t Cnt>
class expr_label_builder_contract;

template<size_t N, size_t M, size_t K>
class expr_label_builder_contract<N, M, K, 1>;

template<size_t N, size_t M, size_t K>
class expr_label_builder_contract_base {
protected:
	template<typename Expr>
	static size_t next_i(const letter_expr<N + M> &label_c,
		const letter_expr<K> &contr, const Expr &expr, size_t i);

	static const letter &get_letter(const letter_expr<N + M> &label_c,
		const letter_expr<K> &contr, size_t i);

};


/**	\brief Builds a label with which a sub-expression should be evaluated

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K, size_t Cnt>
class expr_label_builder_contract :
	public expr_label_builder_contract_base<N, M, K> {
private:
	size_t m_i;
	expr_label_builder_contract<N, M, K, Cnt - 1> m_sub_builder;
	letter_expr<Cnt> m_label;

public:
	template<typename Expr>
	expr_label_builder_contract(const letter_expr<N + M> &label_c,
		const letter_expr<K> &contr, const Expr &expr);

	size_t get_i() const {
		return m_i;
	}

	const letter_expr<Cnt> &get_label() const {
		return m_label;
	}

};


template<size_t N, size_t M, size_t K>
class expr_label_builder_contract<N, M, K, 1> :
	public expr_label_builder_contract_base<N, M, K> {
private:
	size_t m_i;
	letter_expr<1> m_label;

public:
	template<typename Expr>
	expr_label_builder_contract(const letter_expr<N + M> &label_c,
		const letter_expr<K> &contr, const Expr &expr);

	size_t get_i() const {
		return m_i;
	}

	const letter_expr<1> &get_label() const {
		return m_label;
	}

};


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2,
	size_t NTensor1, size_t NOper1, size_t NTensor2, size_t NOper2>
class evalfunctor_contract {
public:
	static const char *k_clazz; //!< Class name
	static const size_t k_ordera = N + K; //!< Order of the first %tensor
	static const size_t k_orderb = M + K; //!< Order of the second %tensor
	static const size_t k_orderc = N + M; //!< Order of the result

	//!	Contraction expression core type
	typedef core_contract<N, M, K, T, E1, E2> core_t;

	//!	Contraction expression type
	typedef expr<k_orderc, T, core_t> expression_t;

	//!	Expression core type of A
	typedef typename E1::core_t core_a_t;

	//!	Expression core type of B
	typedef typename E2::core_t core_b_t;

	//!	Anonymous evaluator type of A
	typedef anon_eval<k_ordera, T, core_a_t> anon_eval_a_t;

	//!	Anonymous evaluator type of B
	typedef anon_eval<k_orderb, T, core_b_t> anon_eval_b_t;

private:
	expr_label_builder_contract<N, M, K, k_ordera> m_labelbld_a;
	expr_label_builder_contract<M, N, K, k_orderb> m_labelbld_b;
	anon_eval_a_t m_eval_a;
	anon_eval_b_t m_eval_b;

public:
	evalfunctor_contract(
		expression_t &expr, const letter_expr<k_orderc> &label);
	void evaluate() throw(exception);

};


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2,
	size_t NTensor1, size_t NOper1>
class evalfunctor_contract<N, M, K, T, E1, E2, NTensor1, NOper1, 1, 0> {
public:
	static const char *k_clazz; //!< Class name
	static const size_t k_ordera = N + K; //!< Order of the first %tensor
	static const size_t k_orderb = M + K; //!< Order of the second %tensor
	static const size_t k_orderc = N + M; //!< Order of the result

	//!	Contraction expression core type
	typedef core_contract<N, M, K, T, E1, E2> core_t;

	//!	Contraction expression type
	typedef expr<k_orderc, T, core_t> expression_t;

	//!	Expression core type of A
	typedef typename E1::core_t core_a_t;

	//!	Expression core type of B
	typedef typename E2::core_t core_b_t;

	//!	Anonymous evaluator type of A
	typedef anon_eval<k_ordera, T, core_a_t> anon_eval_a_t;

private:
	expr_label_builder_contract<N, M, K, k_ordera> m_labelbld_a;
	anon_eval_a_t m_eval_a;

public:
	evalfunctor_contract(
		expression_t &expr, const letter_expr<k_orderc> &label);
	void evaluate() throw(exception);

};


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2,
	size_t NTensor2, size_t NOper2>
class evalfunctor_contract<N, M, K, T, E1, E2, 1, 0, NTensor2, NOper2> {
public:
	static const char *k_clazz; //!< Class name
	static const size_t k_ordera = N + K; //!< Order of the first %tensor
	static const size_t k_orderb = M + K; //!< Order of the second %tensor
	static const size_t k_orderc = N + M; //!< Order of the result

	//!	Contraction expression core type
	typedef core_contract<N, M, K, T, E1, E2> core_t;

	//!	Contraction expression type
	typedef expr<k_orderc, T, core_t> expression_t;

	//!	Expression core type of A
	typedef typename E1::core_t core_a_t;

	//!	Expression core type of B
	typedef typename E2::core_t core_b_t;

	//!	Anonymous evaluator type of B
	typedef anon_eval<k_orderb, T, core_b_t> anon_eval_b_t;

private:
	expr_label_builder_contract<M, N, K, k_orderb> m_labelbld_b;
	anon_eval_b_t m_eval_b;

public:
	evalfunctor_contract(
		expression_t &expr, const letter_expr<k_orderc> &label);
	void evaluate() throw(exception);

};


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
class evalfunctor_contract<N, M, K, T, E1, E2, 1, 0, 1, 0> {
public:
	static const char *k_clazz; //!< Class name
	static const size_t k_ordera = N + K; //!< Order of the first %tensor
	static const size_t k_orderb = M + K; //!< Order of the second %tensor
	static const size_t k_orderc = N + M; //!< Order of the result

	//!	Contraction expression core type
	typedef core_contract<N, M, K, T, E1, E2> core_t;

	//!	Contraction expression type
	typedef expr<k_orderc, T, core_t> expression_t;

private:
	contraction2<N, M, K> m_contr; //!< Contraction

public:
	evalfunctor_contract(
		expression_t &expr, const letter_expr<k_orderc> &label) { }
	void evaluate() throw(exception);

};


template<size_t N, size_t M, size_t K> template<typename Expr>
size_t expr_label_builder_contract_base<N, M, K>::next_i(
	const letter_expr<N + M> &label_c, const letter_expr<K> &contr,
	const Expr &expr, size_t i) {

	if(i < N + M) {
		size_t j = i;
		for(; j < N + M; j++) {
			if(expr.contains(label_c.letter_at(j))) return j;
		}
		for(j = 0; j < K; j++) {
			if(expr.contains(contr.letter_at(j)))
				return j + N + M;
		}
	} else {
		size_t j = i - N - M;
		for(; j < K; j++) {
			if(expr.contains(contr.letter_at(j)))
				return j + N + M;
		}
	}
	throw_exc("expr_label_builder_contract_base", "next_i()",
		"Inconsistent expression");
}


template<size_t N, size_t M, size_t K>
inline const letter &expr_label_builder_contract_base<N, M, K>::get_letter(
	const letter_expr<N + M> &label_c, const letter_expr<K> &contr,
	size_t i) {

	if(i < N + M) return label_c.letter_at(i);
	else return contr.letter_at(i - N - M);
}


template<size_t N, size_t M, size_t K, size_t Cur> template<typename Expr>
expr_label_builder_contract<N, M, K, Cur>::expr_label_builder_contract(
	const letter_expr<N + M> &label_c, const letter_expr<K> &contr,
	const Expr &expr) :
	m_sub_builder(label_c, contr, expr),
	m_i(next_i(label_c, contr, expr, m_sub_builder.get_i() + 1)),
	m_label(m_sub_builder.get_label(), get_letter(label_c, contr, m_i)) {

}


template<size_t N, size_t M, size_t K> template<typename Expr>
expr_label_builder_contract<N, M, K, 1>::expr_label_builder_contract(
	const letter_expr<N + M> &label_c, const letter_expr<K> &contr,
	const Expr &expr) :
	m_i(next_i(label_c, contr, expr, 0)),
	m_label(get_letter(label_c, contr, m_i)) {

}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2,
	size_t NTensor1, size_t NOper1, size_t NTensor2, size_t NOper2>
const char *evalfunctor_contract<N, M, K, T, E1, E2,
	NTensor1, NOper1, NTensor2, NOper2>::k_clazz =
		"evalfunctor_contract<N, M, K, T, E1, E2, "
		"NTensor1, NOper1, NTensor2, NOper2>";


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2,
	size_t NTensor1, size_t NOper1, size_t NTensor2, size_t NOper2>
evalfunctor_contract<N, M, K, T, E1, E2, NTensor1, NOper1, NTensor2,NOper2>::
	evalfunctor_contract(expression_t &expr,
	const letter_expr<k_orderc> &label) :
		m_labelbld_a(label, expr.get_core().get_contr(),
			expr.get_core().get_expr_1()),
		m_labelbld_b(label, expr.get_core().get_contr(),
			expr.get_core().get_expr_2()),
		m_eval_a(expr.get_core().get_expr_1(),
			m_labelbld_a.get_label()),
		m_eval_b(expr.get_core().get_expr_2(),
			m_labelbld_b.get_label()) {


}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2,
	size_t NTensor1, size_t NOper1, size_t NTensor2, size_t NOper2>
void evalfunctor_contract<N, M, K, T, E1, E2,
	NTensor1, NOper1, NTensor2, NOper2>::evaluate() throw(exception) {

	throw_exc(k_clazz, "evaluate()", "NIY");
}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2,
	size_t NTensor1, size_t NOper1>
const char *evalfunctor_contract<N, M, K, T, E1, E2,
	NTensor1, NOper1, 1, 0>::k_clazz =
		"evalfunctor_contract<N, M, K, T, E1, E2, "
		"NTensor1, NOper1, 1, 0>";


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2,
	size_t NTensor1, size_t NOper1>
evalfunctor_contract<N, M, K, T, E1, E2, NTensor1, NOper1, 1, 0>::
	evalfunctor_contract(expression_t &expr,
	const letter_expr<k_orderc> &label) :
		m_labelbld_a(label, expr.get_core().get_contr(),
			expr.get_core().get_expr_1()),
		m_eval_a(expr.get_core().get_expr_1(),
			m_labelbld_a.get_label()) {

}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2,
	size_t NTensor1, size_t NOper1>
void evalfunctor_contract<N, M, K, T, E1, E2,NTensor1, NOper1, 1, 0>::
	evaluate() throw(exception) {

	throw_exc(k_clazz, "evaluate()", "NIY");
}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2,
	size_t NTensor2, size_t NOper2>
const char *evalfunctor_contract<N, M, K, T, E1, E2,
	1, 0, NTensor2, NOper2>::k_clazz =
		"evalfunctor_contract<N, M, K, T, E1, E2, "
		"1, 0, NTensor2, NOper2>";


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2,
	size_t NTensor2, size_t NOper2>
evalfunctor_contract<N, M, K, T, E1, E2, 1, 0, NTensor2, NOper2>::
evalfunctor_contract(expression_t &expr,
	const letter_expr<k_orderc> &label) :
		m_labelbld_b(label, expr.get_core().get_contr(),
			expr.get_core().get_expr_2()),
		m_eval_b(expr.get_core().get_expr_2(),
			m_labelbld_b.get_label()) {

}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2,
	size_t NTensor2, size_t NOper2>
void evalfunctor_contract<N, M, K, T, E1, E2, 1, 0, NTensor2, NOper2>::
	evaluate() throw(exception) {

	throw_exc(k_clazz, "evaluate()", "NIY");
}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
const char *evalfunctor_contract<N, M, K, T, E1, E2, 1, 0, 1, 0>::k_clazz =
		"evalfunctor_contract<N, M, K, T, E1, E2, 1, 0, 1, 0>";


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
void evalfunctor_contract<N, M, K, T, E1, E2, 1, 0, 1, 0>::evaluate()
	throw(exception) {

	throw_exc(k_clazz, "evaluate()", "NIY");
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_EVALFUNCTOR_CONTRACT_H
