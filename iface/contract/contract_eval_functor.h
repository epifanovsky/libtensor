#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_EVAL_FUNCTOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_EVAL_FUNCTOR_H

#include "btod/btod_contract2.h"
#include "iface/expr/anon_eval.h"
#include "core_contract.h"
#include "contract_subexpr_labels.h"

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
class contract_eval_functor_base {
public:
	static const size_t k_ordera = N + K; //!< Order of the first %tensor
	static const size_t k_orderb = M + K; //!< Order of the second %tensor
	static const size_t k_orderc = N + M; //!< Order of the result

protected:
	static contraction2<N, M, K> mk_contr(
		const letter_expr<k_ordera> &label_a,
		const letter_expr<k_orderb> &label_b,
		const letter_expr<k_orderc> &label_c,
		const letter_expr<K> &contr);
};


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2,
	size_t NT1, size_t NO1, size_t NT2, size_t NO2>
class contract_eval_functor :
	public contract_eval_functor_base<N, M, K, T, E1, E2> {
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

	//!	Sub-expression labels
	typedef contract_subexpr_labels<N, M, K, T, E1, E2> subexpr_labels_t;

private:
	anon_eval_a_t m_eval_a; //!< Anonymous evaluator for sub-expression A
	anon_eval_b_t m_eval_b; //!< Anonymous evaluator for sub-expression B

public:
	contract_eval_functor(expression_t &expr,
		const subexpr_labels_t &labels_ab,
		const letter_expr<k_orderc> &label_c);

	void evaluate();

	arg<k_orderc, T, oper_tag> get_arg() const {
		throw_exc(k_clazz, "get_arg()", "NIY");
	}

};


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2,
	size_t NT1, size_t NO1, size_t NT2, size_t NO2>
const char *contract_eval_functor<N, M, K, T, E1, E2, NT1, NO1, NT2, NO2>::
k_clazz = "contract_eval_functor<N, M, K, T, E1, E2, NT1, NO1, NT2, NO2>";


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2,
	size_t NT1, size_t NO1, size_t NT2, size_t NO2>
contract_eval_functor<N, M, K, T, E1, E2, NT1, NO1, NT2, NO2>::
contract_eval_functor(expression_t &expr, const subexpr_labels_t &labels_ab,
	const letter_expr<k_orderc> &label_c) :

	m_eval_a(expr.get_core().get_expr_1(), labels_ab.get_label_a()),
	m_eval_b(expr.get_core().get_expr_2(), labels_ab.get_label_b()) {

}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2,
size_t NT1, size_t NO1, size_t NT2, size_t NO2>
void contract_eval_functor<N, M, K, T, E1, E2, NT1, NO1, NT2, NO2>::evaluate() {

	m_eval_a.evaluate();
	m_eval_b.evaluate();
}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
contraction2<N, M, K> contract_eval_functor_base<N, M, K, T, E1, E2>::mk_contr(
	const letter_expr<k_ordera> &label_a,
	const letter_expr<k_orderb> &label_b,
	const letter_expr<k_orderc> &label_c, const letter_expr<K> &contr) {

	size_t contr_a[K], contr_b[K];
	size_t seq1[k_orderc], seq2[k_orderc];

	for(size_t i = 0; i < k_orderc; i++) seq1[i] = i;

	size_t j = 0, k = 0;
	for(size_t i = 0; i < k_ordera; i++) {
		const letter &l = label_a.letter_at(i);
		if(label_c.contains(l)) {
			seq2[j] = label_c.index_of(l);
			j++;
		} else {
			if(!contr.contains(l)) {
				// throw exception
			}
			contr_a[k] = i;
			contr_b[k] = label_b.index_of(l);
			k++;
		}
	}
	for(size_t i = 0; i < k_orderb; i++) {
		const letter &l = label_b.letter_at(i);
		if(label_c.contains(l)) {
			seq2[j] = label_c.index_of(l);
			j++;
		}
	}

	permutation_builder<k_orderc> permc(seq1, seq2);
	contraction2<N, M, K> c(permc.get_perm());

	for(size_t i = 0; i < K; i++) {
		c.contract(contr_a[i], contr_b[i]);
	}

	return c;
}


} // namespace labeled_btensor_expr
} // namespace libtensor

// Template specializations
#include "contract_eval_functor_xx10.h"
#include "contract_eval_functor_10xx.h"
#include "contract_eval_functor_1010.h"

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_EVAL_FUNCTOR_H
