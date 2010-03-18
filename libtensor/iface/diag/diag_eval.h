#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_EVAL_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_EVAL_H

#include "diag_eval_functor.h"
#include "diag_subexpr_label_builder.h"

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Evaluating container for the extraction of a diagonal
	\tparam N Tensor order.
	\tparam M Diagonal order.
	\tparam E1 Sub-expression core type.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T, typename E1>
class diag_eval {
public:
	static const char *k_clazz; //!< Class name

	//!	Expression core type
	typedef diag_core<N, M, T, E1> core_t;

	//!	Expression type
	typedef expr<N - M + 1, T, core_t> expression_t;

	//!	Evaluating container type of the sub-expression
	typedef typename E1::eval_container_t eval_container_a_t;

	//!	Number of %tensor arguments in the sub-expression
	static const size_t k_narg_tensor_a =
		eval_container_a_t::template narg<tensor_tag>::k_narg;

	//!	Number of operation arguments in the sub-expression
	static const size_t k_narg_oper_a =
		eval_container_a_t::template narg<oper_tag>::k_narg;

	//!	Labels for sub-expressions
	typedef diag_subexpr_label_builder<N, M> subexpr_label_t;

	//!	Evaluating functor type (specialized for the sub-expression)
	typedef diag_eval_functor<N, M, T, E1, k_narg_tensor_a,
		k_narg_oper_a> functor_t;

	//!	Number of arguments in the expression
	template<typename Tag, int Dummy = 0>
	struct narg {
		static const size_t k_narg = 0;
	};

private:
	subexpr_label_t m_sub_label; //!< Sub-expression label
	functor_t m_func; //!< Specialized evaluation functor

public:
	/**	\brief Initializes the container with given expression and
			result recipient
	 **/
	diag_eval(expression_t &expr, const letter_expr<N - M + 1> &label)
		throw(exception);

	/**	\brief Evaluates sub-expressions into temporary tensors
	 **/
	void prepare() throw(exception);

	template<typename Tag>
	arg<N - M + 1, T, Tag> get_arg(const Tag &tag, size_t i) const
		throw(exception);

	/**	\brief Returns tensor arguments
	 **/
	arg<N - M + 1, T, oper_tag> get_arg(const oper_tag &tag,
		size_t i) const throw(exception);

};


template<size_t N, size_t M, typename T, typename E1>
const char *diag_eval<N, M, T, E1>::k_clazz = "diag_eval<N, M, T, E1>";


template<size_t N, size_t M, typename T, typename E1>
template<int Dummy>
struct diag_eval<N, M, T, E1>::narg<oper_tag, Dummy> {
	static const size_t k_narg = 1;
};


template<size_t N, size_t M, typename T, typename E1>
diag_eval<N, M, T, E1>::diag_eval(expression_t &expr,
	const letter_expr<N - M + 1> &label) throw(exception) :

	m_sub_label(label, expr.get_core().get_diag_letter(),
		expr.get_core().get_diag_label()),
	m_func(expr, m_sub_label, label) {

}


template<size_t N, size_t M, typename T, typename E1>
void diag_eval<N, M, T, E1>::prepare() throw(exception) {

	m_func.evaluate();
}


template<size_t N, size_t M, typename T, typename E1>
template<typename Tag>
arg<N - M + 1, T, Tag> diag_eval<N, M, T, E1>::get_arg(
	const Tag &tag, size_t i) const throw(exception) {

	static const char *method = "get_arg(const Tag&, size_t)";

	throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
		"Invalid method.");
}


template<size_t N, size_t M, typename T, typename E1>
arg<N - M + 1, T, oper_tag> diag_eval<N, M, T, E1>::get_arg(
	const oper_tag &tag, size_t i) const throw(exception) {

	static const char *method = "get_arg(const oper_tag&, size_t)";

	if(i != 0) {
		throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
			"i");
	}

	return m_func.get_arg();
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_EVAL_H
