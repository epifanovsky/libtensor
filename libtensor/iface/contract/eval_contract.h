#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_EVAL_CONTRACT_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_EVAL_CONTRACT_H

#include "contract_eval_functor.h"
#include "contract_subexpr_labels.h"

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Evaluating container for the contraction of two tensors
	\tparam N Order of the first %tensor (A) less contraction degree.
	\tparam M Order of the second %tensor (B) less contraction degree.
	\tparam K Number of indexes contracted.
	\tparam Expr1 First expression (A) type.
	\tparam Expr2 Second expression (B) type.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
class eval_contract {
public:
	static const char *k_clazz; //!< Class name
	static const size_t k_ordera = N + K; //!< Order of the first %tensor
	static const size_t k_orderb = M + K; //!< Order of the second %tensor
	static const size_t k_orderc = N + M; //!< Order of the result

	//!	Contraction expression core type
	typedef core_contract<N, M, K, T, E1, E2> core_t;

	//!	Contraction expression type
	typedef expr<k_orderc, T, core_t> expression_t;

	//!	Evaluating container type of the first expression (A)
	typedef typename E1::eval_container_t eval_container_a_t;

	//!	Evaluating container type of the second expression (B)
	typedef typename E2::eval_container_t eval_container_b_t;

	//!	Number of %tensor arguments in expression A
	static const size_t k_narg_tensor_a =
		eval_container_a_t::template narg<tensor_tag>::k_narg;

	//!	Number of operation arguments in expression A
	static const size_t k_narg_oper_a =
		eval_container_a_t::template narg<oper_tag>::k_narg;

	//!	Number of %tensor arguments in expression B
	static const size_t k_narg_tensor_b =
		eval_container_b_t::template narg<tensor_tag>::k_narg;

	//!	Number of operation arguments in expression A
	static const size_t k_narg_oper_b =
		eval_container_b_t::template narg<oper_tag>::k_narg;

	//!	Labels for sub-expressions
	typedef contract_subexpr_labels<N, M, K, T, E1, E2> subexpr_labels_t;

	//!	Evaluating functor type (specialized for A and B)
	typedef contract_eval_functor<N, M, K, T, E1, E2,
		k_narg_tensor_a, k_narg_oper_a, k_narg_tensor_b, k_narg_oper_b>
		functor_t;

	//!	Number of arguments in the expression
	template<typename Tag, int Dummy = 0>
	struct narg {
		static const size_t k_narg = 0;
	};

private:
	subexpr_labels_t m_sub_labels;
	functor_t m_func; //!< Sub-expression evaluation functor


public:
	/**	\brief Initializes the container with given expression and
			result recipient
	 **/
	eval_contract(expression_t &expr, const letter_expr<k_orderc> &label)
		throw(exception);

	/**	\brief Evaluates sub-expressions into temporary tensors
	 **/
	void prepare() throw(exception);

	template<typename Tag>
	arg<N + M, T, Tag> get_arg(const Tag &tag, size_t i) const
		throw(exception);

	/**	\brief Returns a single argument
	 **/
	arg<N + M, T, oper_tag> get_arg(const oper_tag &tag, size_t i) const
		throw(exception);

};


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
const char *eval_contract<N, M, K, T, E1, E2>::k_clazz =
	"eval_contract<N, M, K, T, E1, E2>";


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
template<int Dummy>
struct eval_contract<N, M, K, T, E1, E2>::narg<oper_tag, Dummy> {
	static const size_t k_narg = 1;
};


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
inline eval_contract<N, M, K, T, E1, E2>::eval_contract(
	expression_t &expr, const letter_expr<k_orderc> &label)
	throw(exception) :

	m_sub_labels(expr, label),
	m_func(expr, m_sub_labels, label) {

}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
inline void eval_contract<N, M, K, T, E1, E2>::prepare() throw(exception) {

	m_func.evaluate();
}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
template<typename Tag>
arg<N + M, T, Tag> eval_contract<N, M, K, T, E1, E2>::get_arg(
	const Tag &tag, size_t i) const throw(exception) {

	static const char *method = "get_arg(const Tag&, size_t)";
	throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
		"Invalid method.");
}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
arg<N + M, T, oper_tag> eval_contract<N, M, K, T, E1, E2>::get_arg(
	const oper_tag &tag, size_t i) const throw(exception) {

	static const char *method = "get_arg(const oper_tag&, size_t)";

	if(i != 0) {
		throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Argument index is out of bounds.");
	}

	return m_func.get_arg();
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_EVAL_CONTRACT_H
