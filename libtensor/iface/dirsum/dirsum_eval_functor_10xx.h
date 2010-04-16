#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIRSUM_EVAL_FUNCTOR_10XX_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIRSUM_EVAL_FUNCTOR_10XX_H

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, size_t M, typename T, typename E1, typename E2,
	size_t NT1, size_t NO1, size_t NT2, size_t NO2>
class dirsum_eval_functor;


/**	\brief Functor for evaluating direct sums (tensor + expression)

	\ingroup libtensor_iface
 **/
template<size_t N, size_t M, typename T, typename E1, typename E2,
	size_t NT2, size_t NO2>
class dirsum_eval_functor<N, M, T, E1, E2, 1, 0, NT2, NO2> {
public:
	static const char *k_clazz; //!< Class name
	static const size_t k_ordera = N; //!< Order of the first %tensor
	static const size_t k_orderb = M; //!< Order of the second %tensor
	static const size_t k_orderc = N + M; //!< Order of the result

	//!	Expression core type
	typedef core_dirsum<N, M, T, E1, E2> core_t;

	//!	Expression type
	typedef expr<k_orderc, T, core_t> expression_t;

	//!	Expression core type of A
	typedef typename E1::core_t core_a_t;

	//!	Expression core type of B
	typedef typename E2::core_t core_b_t;

	//!	Evaluating container type of A
	typedef typename expr<k_ordera, T, core_a_t>::eval_container_t
		eval_container_a_t;

	//!	Anonymous evaluator type of B
	typedef anon_eval<k_orderb, T, core_b_t> anon_eval_b_t;

	//!	Sub-expressions labels
	typedef dirsum_subexpr_labels<N, M, T, E1, E2> subexpr_labels_t;

private:
	eval_container_a_t m_eval_a; //!< Container for tensor A
	arg<k_ordera, T, tensor_tag> m_arg_a; //!< Tensor argument for A
	permutation<k_ordera> m_invperm_a;
	anon_eval_b_t m_eval_b; //!< Anonymous evaluator for sub-expression B
	permutation<k_orderb> m_invperm_b;
	dirsum_permutation_builder<N, M> m_perm_bld; //!< Permutation builder
	btod_dirsum<N, M> m_op; //!< Direct sum operation
	arg<k_orderc, T, oper_tag> m_arg; //!< Composed operation argument

public:
	dirsum_eval_functor(expression_t &expr,
		const subexpr_labels_t &labels_ab,
		const letter_expr<k_orderc> &label_c);

	void evaluate();

	void clean();

	arg<N + M, T, oper_tag> get_arg() const { return m_arg; }

};


template<size_t N, size_t M, typename T, typename E1, typename E2,
	size_t NT2, size_t NO2>
const char *dirsum_eval_functor<N, M, T, E1, E2, 1, 0, NT2, NO2>::
	k_clazz = "dirsum_eval_functor<N, M, T, E1, E2, 1, 0, NT2, NO2>";


template<size_t N, size_t M, typename T, typename E1, typename E2,
	size_t NT2, size_t NO2>
dirsum_eval_functor<N, M, T, E1, E2, 1, 0, NT2, NO2>::
dirsum_eval_functor(expression_t &expr, const subexpr_labels_t &labels_ab,
	const letter_expr<k_orderc> &label_c) :

	m_eval_a(expr.get_core().get_expr_1(), labels_ab.get_label_a()),
	m_arg_a(m_eval_a.get_arg(tensor_tag(), 0)),
	m_invperm_a(m_arg_a.get_perm(), true),
	m_eval_b(expr.get_core().get_expr_2(), labels_ab.get_label_b()),
	m_perm_bld(labels_ab.get_label_a(), m_invperm_a,
		labels_ab.get_label_b(), m_invperm_b, label_c),
	m_op(m_arg_a.get_btensor(), m_arg_a.get_coeff(),
		m_eval_b.get_btensor(), 1.0, m_perm_bld.get_perm()),
	m_arg(m_op, 1.0) {

}


template<size_t N, size_t M, typename T, typename E1, typename E2,
	size_t NT2, size_t NO2>
void dirsum_eval_functor<N, M, T, E1, E2, 1, 0, NT2, NO2>::evaluate() {

	m_eval_b.evaluate();
}


template<size_t N, size_t M, typename T, typename E1, typename E2,
	size_t NT2, size_t NO2>
void dirsum_eval_functor<N, M, T, E1, E2, 1, 0, NT2, NO2>::clean() {

	m_eval_b.clean();
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIRSUM_EVAL_FUNCTOR_10XX_H
