#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_EVAL_FUNCTOR_10_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_EVAL_FUNCTOR_10_H

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, size_t M, typename T, typename E1, size_t NT1, size_t NO1>
class diag_eval_functor;


/**	\brief Functor for evaluating the diagonal (tensor + tensor)

	\ingroup libtensor_iface
 **/
template<size_t N, size_t M, typename T, typename E1>
class diag_eval_functor<N, M, T, E1, 1, 0> {
public:
	static const char *k_clazz; //!< Class name
	static const size_t k_ordera = N; //!< Order of the RHS
	static const size_t k_orderb = N - M + 1; //!< Order of the LHS

	//!	Expression core type
	typedef core_diag<N, M, T, E1> core_t;

	//!	Expression type
	typedef expr<k_orderb, T, core_t> expression_t;

	//!	Evaluating container type of the sub-expression
	typedef typename E1::eval_container_t eval_container_a_t;

	//!	Sub-expressions labels
	typedef contract_subexpr_labels<N, M, K, T, E1, E2> subexpr_labels_t;

private:
	eval_container_a_t m_eval_a; //!< Container for tensor A
	arg<k_ordera, T, tensor_tag> m_arg_a; //!< Tensor argument for A
	permutation<k_ordera> m_invperm_a; //!< Permutation of A
	diag_mask_builder<N, M> m_mask_bld; //!< Diagonal %mask builder
	btod_diag<N, M> m_op; //!< Diagonal extraction operation
	arg<k_orderc, T, oper_tag> m_arg; //!< Composed operation argument

public:
	diag_eval_functor(expression_t &expr,
		const subexpr_labels_t &labels_ab,
		const letter_expr<k_orderc> &label_c);

	void evaluate() throw(exception) { }

	arg<N - M + 1, T, oper_tag> get_arg() const { return m_arg; }

};


template<size_t N, size_t M, typename T, typename E1>
const char *diag_eval_functor<N, M, T, E1, 1, 0>::k_clazz =
	"diag_eval_functor<N, M, T, E1, 1, 0>";


template<size_t N, size_t M, typename T, typename E1>
diag_eval_functor<N, M, T, E1, 1, 0>::diag_eval_functor(
	expression_t &expr, const subexpr_labels_t &labels_ab,
	const letter_expr<k_orderc> &label_c) :

	m_eval_a(expr.get_core().get_sub_expr(), labels_ab.get_label_a()),
	m_arg_a(m_eval_a.get_arg(tensor_tag(), 0)),
	m_invperm_a(m_arg_a.get_perm(), true),
	m_mask_bld(),
	m_op(m_arg_a.get_btensor(), m_mask_bld.get_mask(),
		m_mask_bld.get_perm(), m_arg_a.get_coeff()),
	m_arg(m_op, 1.0) {

}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_EVAL_FUNCTOR_10_H
