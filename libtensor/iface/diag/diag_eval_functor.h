#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_EVAL_FUNCTOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_EVAL_FUNCTOR_H

#include "../../btod/btod_diag.h"
#include "../expr/anon_eval.h"
#include "diag_core.h"
#include "diag_subexpr_label_builder.h"
#include "contract_contraction2_builder.h"

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, size_t M, typename T, typename E1, size_t NT1, size_t NO1>
class diag_eval_functor {
public:
	static const char *k_clazz; //!< Class name
	static size_t k_ordera = N; //!< Order of RHS
	static size_t k_orderb = N - M + 1; //!< Order of LHS

	//!	Expression core type
	typedef diag_core<N, M, T, E1> core_t;

	//!	Expression type
	typedef expr<k_orderb, T, core_t> expression_t;


	//!	Expression core type of A
	typedef typename E1::core_t core_a_t;

	//!	Anonymous evaluator type of A
	typedef anon_eval<k_ordera, T, core_a_t> anon_eval_a_t;

	//!	Sub-expression labels
	typedef contract_subexpr_labels<N, M, K, T, E1, E2> subexpr_labels_t;

private:
	anon_eval_a_t m_eval_a; //!< Anonymous evaluator for the sub-expression
	permutation<k_ordera> m_invperm_a;
	contract_contraction2_builder<N, M, K> m_contr_bld; //!< Contraction builder
	btod_diag<N, M> m_op; //!< Diagonal extraction operation
	arg<k_orderc, T, oper_tag> m_arg; //!< Composed operation argument

public:
	diag_eval_functor(expression_t &expr,
		const subexpr_labels_t &labels_ab,
		const letter_expr<k_orderc> &label_c);

	void evaluate();

	arg<N - M + 1, T, oper_tag> get_arg() const { return m_arg; }

};


template<size_t N, size_t M, typename T, typename E1, size_t NT1, size_t NO1>
const char *diag_eval_functor<N, M, T, E1, NT1, NO1>::k_clazz =
	"diag_eval_functor<N, M, T, E1, NT1, NO1>";


template<size_t N, size_t M, typename T, typename E1, size_t NT1, size_t NO1>
diag_eval_functor<N, M, T, E1, NT1, NO1>::diag_eval_functor(
	expression_t &expr, const subexpr_labels_t &labels_ab,
	const letter_expr<k_orderc> &label_c) :

	m_eval_a(expr.get_core().get_expr_1(), labels_ab.get_label_a()),
	m_eval_b(expr.get_core().get_expr_2(), labels_ab.get_label_b()),
	m_contr_bld(labels_ab.get_label_a(), m_invperm_a,
		labels_ab.get_label_b(), m_invperm_b,
		label_c, expr.get_core().get_contr()),
	m_op(m_contr_bld.get_contr(), m_eval_a.get_btensor(), m_eval_b.get_btensor()),
	m_arg(m_op, 1.0) {

}


template<size_t N, size_t M, typename T, typename E1, size_t NT1, size_t NO1>
void diag_eval_functor<N, M, T, E1, NT1, NO1>::evaluate() {

	m_eval_a.evaluate();
	m_eval_b.evaluate();
}


} // namespace labeled_btensor_expr
} // namespace libtensor

// Template specializations
#include "diag_eval_functor_10.h"

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_EVAL_FUNCTOR_H
