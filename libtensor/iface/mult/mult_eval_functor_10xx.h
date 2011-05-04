#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_MULT_EVAL_FUNCTOR_10XX_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_MULT_EVAL_FUNCTOR_10XX_H

namespace libtensor {
namespace labeled_btensor_expr {

template<size_t N, typename T, typename E1, typename E2, bool Recip,
	size_t NT1, size_t NO1, size_t NT2, size_t NO2>
class mult_eval_functor;

/** \brief Function for evaluating element-wise multiplication
		(tensor * expression)

 	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename E1, typename E2, bool Recip,
	size_t NT2, size_t NO2>
class mult_eval_functor<N, T, E1, E2, Recip, 1, 0, NT2, NO2> {
public:
	static const char *k_clazz; //!< Class name

	//!	Multiplication expression core type
	typedef core_mult<N, T, E1, E2, Recip> core_t;

	//!	Multiplication expression type
	typedef expr<N, T, core_t> expression_t;

	//!	Expression core type of A
	typedef typename E1::core_t core_a_t;

	//!	Expression core type of B
	typedef typename E2::core_t core_b_t;

	//!	Evaluation container of first expression (A)
	typedef typename expr<N, T, core_a_t>::eval_container_t eval_container_a_t;

	//!	Anonymous evaluator type of B
	typedef direct_eval<N, T, core_b_t> anon_eval_b_t;

private:
	eval_container_a_t m_eval_a; //!< Container for tensor A
	arg<N, T, tensor_tag> m_arg_a; //!< Tensor argument for A
	anon_eval_b_t m_eval_b; //!< Anonymous evaluator for sub-expression B
	permutation<N> m_perm_b; //!< Permutation of B (dummy)
	btod_mult<N> *m_op; //!< Element-wise multiplication operation
	arg<N, T, oper_tag> *m_arg; //!< Composed operation argument

public:
	mult_eval_functor(expression_t &expr, const letter_expr<N> &label);

	~mult_eval_functor();

	void evaluate();

	void clean();

	arg<N, T, oper_tag> get_arg() const { return *m_arg; }

private:
	void create_arg();
	void destroy_arg();

};


template<size_t N, typename T, typename E1, typename E2, bool Recip,
	size_t NT2, size_t NO2>
const char *mult_eval_functor<N, T, E1, E2, Recip, 1, 0, NT2, NO2>::k_clazz =
		"mult_eval_functor<N, T, Recip, E1, E2, 1, 0, NT2, NO2>";

template<size_t N, typename T, typename E1, typename E2, bool Recip,
	size_t NT2, size_t NO2>
mult_eval_functor<N, T, E1, E2, Recip, 1, 0, NT2, NO2>::mult_eval_functor(
		expression_t &expr, const letter_expr<N> &label) :

	m_eval_a(expr.get_core().get_expr_1(), label),
	m_arg_a(m_eval_a.get_arg(tensor_tag(), 0)),
	m_eval_b(expr.get_core().get_expr_2(), label),
	m_op(0), m_arg(0) {

}


template<size_t N, typename T, typename E1, typename E2, bool Recip,
	size_t NT2, size_t NO2>
mult_eval_functor<N, T, E1, E2, Recip, 1, 0, NT2, NO2>::~mult_eval_functor() {

	destroy_arg();
}


template<size_t N, typename T, typename E1, typename E2, bool Recip,
	size_t NT2, size_t NO2>
void mult_eval_functor<N, T, E1, E2, Recip, 1, 0, NT2, NO2>::evaluate() {

	m_eval_b.evaluate();
	create_arg();
}


template<size_t N, typename T, typename E1, typename E2, bool Recip,
	size_t NT2, size_t NO2>
void mult_eval_functor<N, T, E1, E2, Recip, 1, 0, NT2, NO2>::clean() {

	destroy_arg();
	m_eval_b.clean();
}


template<size_t N, typename T, typename E1, typename E2, bool Recip,
	size_t NT2, size_t NO2>
void mult_eval_functor<N, T, E1, E2, Recip, 1, 0, NT2, NO2>::create_arg() {

	destroy_arg();
	m_op = new btod_mult<N>(m_arg_a.get_btensor(), m_arg_a.get_perm(),
			m_eval_b.get_btensor(), m_perm_b, Recip, m_arg_a.get_coeff());
	m_arg = new arg<N, T, oper_tag>(*m_op, 1.0);
}


template<size_t N, typename T, typename E1, typename E2, bool Recip,
	size_t NT2, size_t NO2>
void mult_eval_functor<N, T, E1, E2, Recip, 1, 0, NT2, NO2>::destroy_arg() {

	delete m_arg; m_arg = 0;
	delete m_op; m_op = 0;
}


} // namespace labeled_btensor_expr
} // namespace libtensor


#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_MULT_EVAL_FUNCTOR_10XX_H
