#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_EVAL_FUNCTOR_10_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_EVAL_FUNCTOR_10_H

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, size_t M, typename T, typename E1, size_t NT1, size_t NO1>
class diag_eval_functor;


/**	\brief Functor for evaluating the diagonal (tensor)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T, typename E1>
class diag_eval_functor<N, M, T, E1, 1, 0> {
public:
	static const char *k_clazz; //!< Class name
	static const size_t k_ordera = N; //!< Order of the RHS
	static const size_t k_orderb = N - M + 1; //!< Order of the LHS

	//!	Expression core type
	typedef diag_core<N, M, T, E1> core_t;

	//!	Expression type
	typedef expr<k_orderb, T, core_t> expression_t;

	//!	Evaluating container type of the sub-expression
	typedef typename E1::eval_container_t eval_container_a_t;

	//!	Sub-expressions labels
	typedef diag_subexpr_label_builder<N, M> subexpr_label_t;

private:
	eval_container_a_t m_eval_a; //!< Container for tensor A
	arg<k_ordera, T, tensor_tag> m_arg_a; //!< Tensor argument for A
	permutation<k_ordera> m_invperm_a; //!< Permutation of A
	diag_params_builder<N, M> m_params_bld; //!< Parameters builder
	btod_diag<N, M> m_op; //!< Diagonal extraction operation
	arg<k_orderb, T, oper_tag> m_arg; //!< Composed operation argument

public:
	diag_eval_functor(expression_t &expr,
		const subexpr_label_t &label_a,
		const letter_expr<k_orderb> &label_b);

	void evaluate() { }

	void clean() { }

	arg<N - M + 1, T, oper_tag> get_arg() const { return m_arg; }

};


template<size_t N, size_t M, typename T, typename E1>
const char *diag_eval_functor<N, M, T, E1, 1, 0>::k_clazz =
	"diag_eval_functor<N, M, T, E1, 1, 0>";


template<size_t N, size_t M, typename T, typename E1>
diag_eval_functor<N, M, T, E1, 1, 0>::diag_eval_functor(
	expression_t &expr, const subexpr_label_t &label_a,
	const letter_expr<k_orderb> &label_b) :

	m_eval_a(expr.get_core().get_sub_expr(), label_a.get_label()),
	m_arg_a(m_eval_a.get_arg(tensor_tag(), 0)),
	m_invperm_a(m_arg_a.get_perm(), true),
	m_params_bld(label_a.get_label(), m_invperm_a, label_b,
		expr.get_core().get_diag_letter(),
		expr.get_core().get_diag_label()),
	m_op(m_arg_a.get_btensor(), m_params_bld.get_mask(),
		m_params_bld.get_perm(), m_arg_a.get_coeff()),
	m_arg(m_op, 1.0) {

}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_EVAL_FUNCTOR_10_H
