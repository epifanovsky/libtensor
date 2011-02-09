#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_MULT_EVAL_FUNCTOR_1010_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_MULT_EVAL_FUNCTOR_1010_H

namespace libtensor {
namespace labeled_btensor_expr {

template<size_t N, typename T, typename E1, typename E2, bool Recip,
	size_t NT1, size_t NO1, size_t NT2, size_t NO2>
class mult_eval_functor;

/** \brief Function for evaluating element-wise multiplication (tensor * tensor)

 	\ingroup libtensor_iface
 **/
template<size_t N, typename T, typename E1, typename E2, bool Recip>
class mult_eval_functor<N, T, E1, E2, Recip, 1, 0, 1, 0> {
public:
	static const char *k_clazz; //!< Class name

	//!	Multiplication expression core type
	typedef core_mult<N, T, E1, E2, Recip> core_t;

	//!	Multiplication expression type
	typedef expr<N, T, core_t> expression_t;

	//!	Evaluation conrainer of first expression (A)
	typedef typename E1::eval_container_t eval_container_a_t;

	//!	Evaluation conrainer of second expression (B)
	typedef typename E2::eval_container_t eval_container_b_t;

private:
	eval_container_a_t m_eval_a; //!< Container for tensor A
	arg<N, T, tensor_tag> m_arg_a; //!< Tensor argument for A
	eval_container_b_t m_eval_b; //!< Container for tensor B
	arg<N, T, tensor_tag> m_arg_b; //!< Tensor argument for B

	btod_mult<N> m_op; //!< Multiplication operation
	arg<N, T, oper_tag> m_arg; //!< Composed operation argument

public:
	mult_eval_functor(expression_t &expr, const letter_expr<N> &label);

	~mult_eval_functor() { }

	void evaluate() { }

	void clean() { }

	arg<N, T, oper_tag> get_arg() const { return m_arg; }

};


template<size_t N, typename T, typename E1, typename E2, bool Recip>
const char *mult_eval_functor<N, T, E1, E2, Recip, 1, 0, 1, 0>::k_clazz =
		"mult_eval_functor<N, T, E1, E2, Recip, 1, 0, 1, 0>";

template<size_t N, typename T, typename E1, typename E2, bool Recip>
mult_eval_functor<N, T, E1, E2, Recip, 1, 0, 1, 0>::mult_eval_functor(
		expression_t &expr, const letter_expr<N> &label) :

	m_eval_a(expr.get_core().get_expr_1(), label),
	m_arg_a(m_eval_a.get_arg(tensor_tag(), 0)),
	m_eval_b(expr.get_core().get_expr_2(), label),
	m_arg_b(m_eval_b.get_arg(tensor_tag(), 0)),
	m_op(m_arg_a.get_btensor(), m_arg_a.get_perm(),
			m_arg_b.get_btensor(), m_arg_b.get_perm(), Recip,
			(Recip ? m_arg_a.get_coeff() / m_arg_b.get_coeff()
					: m_arg_a.get_coeff() *  m_arg_b.get_coeff())),
	m_arg(m_op, 1.0) {

}


} // namespace labeled_btensor_expr
} // namespace libtensor


#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_MULT_EVAL_FUNCTOR_1010_H
