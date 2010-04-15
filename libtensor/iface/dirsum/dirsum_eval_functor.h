#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIRSUM_EVAL_FUNCTOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIRSUM_EVAL_FUNCTOR_H

#include "../../btod/btod_dirsum.h"
#include "../expr/anon_eval.h"
#include "core_dirsum.h"
#include "dirsum_permutation_builder.h"
#include "dirsum_subexpr_labels.h"

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, size_t M, typename T, typename E1, typename E2,
	size_t NT1, size_t NO1, size_t NT2, size_t NO2>
class dirsum_eval_functor {
public:
	static const char *k_clazz; //!< Class name
	static const size_t k_ordera = N; //!< Order of the first %tensor
	static const size_t k_orderb = M; //!< Order of the second %tensor
	static const size_t k_orderc = N + M; //!< Order of the result

	//!	Contraction expression core type
	typedef core_dirsum<N, M, T, E1, E2> core_t;

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
	typedef dirsum_subexpr_labels<N, M, T, E1, E2> subexpr_labels_t;

private:
	anon_eval_a_t m_eval_a; //!< Anonymous evaluator for sub-expression A
	permutation<k_ordera> m_invperm_a; //!< Identity perm for A
	anon_eval_b_t m_eval_b; //!< Anonymous evaluator for sub-expression B
	permutation<k_orderb> m_invperm_b; //!< Identity perm for B
	dirsum_permutation_builder<N, M> m_perm_bld;
	btod_dirsum<N, M> m_op; //!< Direct sum operation
	arg<k_orderc, T, oper_tag> m_arg; //!< Composed operation argument

public:
	dirsum_eval_functor(expression_t &expr,
		const subexpr_labels_t &labels_ab,
		const letter_expr<k_orderc> &label_c);

	void evaluate();

	void clean();

	arg<k_orderc, T, oper_tag> get_arg() const { return m_arg; }

};


template<size_t N, size_t M, typename T, typename E1, typename E2,
	size_t NT1, size_t NO1, size_t NT2, size_t NO2>
const char *dirsum_eval_functor<N, M, T, E1, E2, NT1, NO1, NT2, NO2>::
k_clazz = "dirsum_eval_functor<N, M, T, E1, E2, NT1, NO1, NT2, NO2>";


template<size_t N, size_t M, typename T, typename E1, typename E2,
	size_t NT1, size_t NO1, size_t NT2, size_t NO2>
dirsum_eval_functor<N, M, T, E1, E2, NT1, NO1, NT2, NO2>::
dirsum_eval_functor(expression_t &expr, const subexpr_labels_t &labels_ab,
	const letter_expr<k_orderc> &label_c) :

	m_eval_a(expr.get_core().get_expr_1(), labels_ab.get_label_a()),
	m_eval_b(expr.get_core().get_expr_2(), labels_ab.get_label_b()),
	m_perm_bld(labels_ab.get_label_a(), m_invperm_a,
		labels_ab.get_label_b(), m_invperm_b, label_c),
	m_op(m_eval_a.get_btensor(), 1.0, m_eval_b.get_btensor(), 1.0,
		m_perm_bld.get_perm()),
	m_arg(m_op, 1.0) {

}


template<size_t N, size_t M, typename T, typename E1, typename E2,
	size_t NT1, size_t NO1, size_t NT2, size_t NO2>
void dirsum_eval_functor<N, M, T, E1, E2, NT1, NO1, NT2, NO2>::evaluate() {

	m_eval_a.evaluate();
	m_eval_b.evaluate();
}


template<size_t N, size_t M, typename T, typename E1, typename E2,
	size_t NT1, size_t NO1, size_t NT2, size_t NO2>
void dirsum_eval_functor<N, M, T, E1, E2, NT1, NO1, NT2, NO2>::clean() {

	m_eval_a.clean();
	m_eval_b.clean();
}


} // namespace labeled_btensor_expr
} // namespace libtensor

// Template specializations
#include "dirsum_eval_functor_xx10.h"
#include "dirsum_eval_functor_10xx.h"
#include "dirsum_eval_functor_1010.h"

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIRSUM_EVAL_FUNCTOR_H
