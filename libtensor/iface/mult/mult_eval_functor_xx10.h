#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_MULT_EVAL_FUNCTOR_XX10_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_MULT_EVAL_FUNCTOR_XX10_H

namespace libtensor {
namespace labeled_btensor_expr {

template<size_t N, typename T, typename E1, typename E2, bool Recip,
	size_t NT1, size_t NO1, size_t NT2, size_t NO2>
class mult_eval_functor;

template<size_t N, typename T, typename E1, typename E2, bool Recip,
	size_t NT1, size_t NO1>
class mult_eval_functor<N, T, E1, E2, Recip, NT1, NO1, 1, 0> {
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

	//!	Anonymous evaluator type of A
	typedef direct_eval<N, T, core_a_t> anon_eval_a_t;

	//!	Evaluation container of second expression (B)
	typedef typename expr<N, T, core_b_t>::eval_container_t eval_container_b_t;

private:
	anon_eval_a_t m_eval_a; //!< Anonymous evaluator for sub-expression A
	permutation<N> m_perm_a; //!< Permutation of A (dummy)
	eval_container_b_t m_eval_b; //!< Container for tensor B
	arg<N, T, tensor_tag> m_arg_b; //!< Tensor argument for B

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
	size_t NT1, size_t NO1>
const char *mult_eval_functor<N, T, E1, E2, Recip, NT1, NO1, 1, 0>::k_clazz =
		"mult_eval_functor<N, T, E1, E2, Recip, NT1, NO1, 1, 0>";

template<size_t N, typename T, typename E1, typename E2, bool Recip,
	size_t NT1, size_t NO1>
mult_eval_functor<N, T, E1, E2, Recip, NT1, NO1, 1, 0>::mult_eval_functor(
		expression_t &expr, const letter_expr<N> &label) :

	m_eval_a(expr.get_core().get_expr_1(), label),
	m_eval_b(expr.get_core().get_expr_2(), label),
	m_arg_b(m_eval_b.get_arg(tensor_tag(), 0)),
	m_op(0), m_arg(0) {

}


template<size_t N, typename T, typename E1, typename E2, bool Recip,
	size_t NT1, size_t NO1>
mult_eval_functor<N, T, E1, E2, Recip, NT1, NO1, 1, 0>::~mult_eval_functor() {

	destroy_arg();
}


template<size_t N, typename T, typename E1, typename E2, bool Recip,
	size_t NT1, size_t NO1>
void mult_eval_functor<N, T, E1, E2, Recip, NT1, NO1, 1, 0>::evaluate() {

	m_eval_a.evaluate();
	create_arg();
}


template<size_t N, typename T, typename E1, typename E2, bool Recip,
	size_t NT1, size_t NO1>
void mult_eval_functor<N, T, E1, E2, Recip, NT1, NO1, 1, 0>::clean() {

	destroy_arg();
	m_eval_a.clean();
}


template<size_t N, typename T, typename E1, typename E2, bool Recip,
	size_t NT1, size_t NO1>
void mult_eval_functor<N, T, E1, E2, Recip, NT1, NO1, 1, 0>::create_arg() {

	destroy_arg();
	m_op = new btod_mult<N>(m_eval_a.get_btensor(), m_perm_a,
			m_arg_b.get_btensor(), m_arg_b.get_perm(), Recip,
			(Recip ? 1. / m_arg_b.get_coeff() : m_arg_b.get_coeff()));
	m_arg = new arg<N, T, oper_tag>(*m_op, 1.0);
}


template<size_t N, typename T, typename E1, typename E2, bool Recip,
	size_t NT1, size_t NO1>
void mult_eval_functor<N, T, E1, E2, Recip, NT1, NO1, 1, 0>::destroy_arg() {

	delete m_arg; m_arg = 0;
	delete m_op; m_op = 0;
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_MULT_EVAL_FUNCTOR_H
